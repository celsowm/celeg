#include "celeg/detail/model/impl.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <cstddef>
#include <cstdint>
namespace celeg {

void Model::Impl::reset(bool allocate_local_kv) {
    allocate_local_kv = allocate_local_kv && options_.allocate_local_kv_cache;
    if (allocate_local_kv && !local_kv_cache_available_) {
        const size_t cache_elements =
            static_cast<size_t>(max_context_) * shape_.kv_width;
        const size_t scale_elements =
            static_cast<size_t>(max_context_) * shape_.num_key_value_heads;
        for (Layer& layer : layers_) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention) continue;
            if (options_.kv_cache_mode == KvCacheMode::Int8) {
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
    position_ = 0;
    phase_ = SessionPhase::Empty;
    const int32_t zero = 0;
    uint64_t seed = generation_.seed;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &zero, sizeof(zero),
                             cudaMemcpyHostToDevice, stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(rng_state_.data(), &seed, sizeof(seed),
                             cudaMemcpyHostToDevice, stream_.get()));
    seen_tokens_.zero_async(stream_.get());
    // Zero convolution state (running buffer that must start empty).  KV
    // caches are intentionally NOT zeroed here: attention only reads
    // positions 0..position_ and every used slot is overwritten before
    // becoming visible after the position reset above.
    for (Layer& layer : layers_) {
        if (AttentionLayer* attention = as_attention(layer)) {
            (void)attention;
            continue;
        }
        as_convolution(layer)->conv_state.zero_async(stream_.get());
    }
    // Prime the FFN-done, router-done and prefetch-done events so the offload
    // transfer stream can start promoting experts on the first layer of the
    // next forward pass.
    CELEG_CUDA(cudaEventRecord(ffn_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(router_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(prefetch_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

void Model::Impl::allocate_prefill_workspace(int rows) {
    const size_t r = static_cast<size_t>(rows);
    prefill_tokens_.reserve(r);
    prefill_hidden_.reserve(r * shape_.hidden);
    prefill_residual_.reserve(r * shape_.hidden);
    prefill_normed_.reserve(r * shape_.hidden);
    prefill_op_output_.reserve(r * shape_.hidden);
    prefill_qkv_.reserve(r * shape_.qkv_width);
    prefill_q_.reserve(r * shape_.q_width);
    prefill_k_.reserve(r * shape_.kv_width);
    prefill_v_.reserve(r * shape_.kv_width);
    prefill_conv_projected_.reserve(r * 3 * shape_.hidden);
    prefill_gate_up_.reserve(r * 2 * shape_.intermediate);
    prefill_activated_.reserve(r * shape_.intermediate);
    prefill_mlp_output_.reserve(r * shape_.hidden);

    if (rows <= kMaxGemmAttentionRows) {
        const size_t scores_elems =
            static_cast<size_t>(shape_.num_attention_heads) * r * r;
        prefill_attn_scores_.reserve(scores_elems);
        prefill_attn_probs_.reserve(scores_elems);
    } else {
        const size_t chunks = (r + kPrefillAttnChunkTokens - 1) / kPrefillAttnChunkTokens;
        const size_t partials = r * shape_.num_attention_heads * chunks;
        prefill_attn_partial_max_.reserve(partials);
        prefill_attn_partial_denom_.reserve(partials);
        prefill_attn_partial_accum_.reserve(partials * shape_.head_dim);
    }
}

void Model::Impl::release_prefill_workspace() {
}

} // namespace celeg

