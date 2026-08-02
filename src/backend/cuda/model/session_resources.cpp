#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <cstddef>
#include <cstdint>
namespace celeg {

void CudaCompiledModel::reset(bool allocate_local_kv) {
    allocate_local_kv = allocate_local_kv && resources_.options_.allocate_local_kv_cache;
    if (allocate_local_kv && !local_kv_cache_available_) {
        const size_t cache_elements =
            static_cast<size_t>(max_context_) * resources_.shape_.kv_width;
        const size_t scale_elements =
            static_cast<size_t>(max_context_) * resources_.shape_.num_key_value_heads;
        for (Layer& layer : resources_.layers_) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention) continue;
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->key_cache_int8.reset(cache_elements);
                attention->value_cache_int8.reset(cache_elements);
                attention->key_cache_scales.reset(scale_elements);
                attention->value_cache_scales.reset(scale_elements);
            } else {
                attention->key_cache.reset(cache_elements);
                attention->value_cache.reset(cache_elements);
            }
        }
        local_kv_cache_available_ = true;
    }
    session_.position_ = 0;
    session_.phase_ = SessionPhase::Empty;
    const int32_t zero = 0;
    uint64_t seed = session_.generation_.seed;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &zero, sizeof(zero),
                             cudaMemcpyHostToDevice, stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(sampling_.rng_state.data(), &seed, sizeof(seed),
                             cudaMemcpyHostToDevice, stream_.get()));
    sampling_.seen_tokens.zero_async(stream_.get());
    // Zero convolution state (running buffer that must start empty).  KV
    // caches are intentionally NOT zeroed here: attention only reads
    // positions 0..position_ and every used slot is overwritten before
    // becoming visible after the position reset above.
    for (Layer& layer : resources_.layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            (void)attention;
            continue;
        }
        as_convolution(layer)->conv_state.zero_async(stream_.get());
    }
    // Prime the FFN-done, router-done and prefetch-done events so the offload
    // transfer stream can start promoting experts on the first layer of the
    // next forward pass.
    CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(workspace_.prefetch_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

void CudaCompiledModel::allocate_prefill_workspace(int rows) {
    const size_t r = static_cast<size_t>(rows);
    workspace_.prefill_tokens_.reserve(r);
    workspace_.prefill_hidden_.reserve(r * resources_.shape_.hidden);
    workspace_.prefill_residual_.reserve(r * resources_.shape_.hidden);
    workspace_.prefill_normed_.reserve(r * resources_.shape_.hidden);
    workspace_.prefill_op_output_.reserve(r * resources_.shape_.hidden);
    workspace_.prefill_qkv_.reserve(r * resources_.shape_.qkv_width);
    workspace_.prefill_q_.reserve(r * resources_.shape_.q_width);
    workspace_.prefill_k_.reserve(r * resources_.shape_.kv_width);
    workspace_.prefill_v_.reserve(r * resources_.shape_.kv_width);
    workspace_.prefill_conv_projected_.reserve(r * 3 * resources_.shape_.hidden);
    workspace_.prefill_gate_up_.reserve(r * 2 * resources_.shape_.intermediate);
    workspace_.prefill_activated_.reserve(r * resources_.shape_.intermediate);
    workspace_.prefill_mlp_output_.reserve(r * resources_.shape_.hidden);

    if (rows <= kMaxGemmAttentionRows) {
        const size_t scores_elems =
            static_cast<size_t>(resources_.shape_.num_attention_heads) * r * r;
        workspace_.prefill_attn_scores_.reserve(scores_elems);
        workspace_.prefill_attn_probs_.reserve(scores_elems);
    } else {
        const size_t chunks = (r + kPrefillAttnChunkTokens - 1) / kPrefillAttnChunkTokens;
        const size_t partials = r * resources_.shape_.num_attention_heads * chunks;
        workspace_.prefill_attn_partial_max_.reserve(partials);
        workspace_.prefill_attn_partial_denom_.reserve(partials);
        workspace_.prefill_attn_partial_accum_.reserve(partials * resources_.shape_.head_dim);
    }
}

void CudaCompiledModel::release_prefill_workspace() {
}

} // namespace celeg

