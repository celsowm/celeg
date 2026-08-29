#include "celeg/backend/cuda/model/detail/compiled_model.hpp"
#include "celeg/backend/cuda/workspace_plan.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
namespace celeg {

void CudaCompiledModel::reset(bool allocate_local_kv) {
    discard_speculative_state();
    mtp_candidate_ready_ = false;
    mtp_candidate_used_ = false;
    ++storage_generation_;
    allocate_local_kv = allocate_local_kv && resources_.options_.allocate_local_kv_cache;
    if (allocate_local_kv && !local_kv_cache_available_) {
        for (Layer& layer : resources_.layers_) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention) continue;
            const AttentionSpec& layout = attention->layout;
            if (layout.uses_latent_state()) {
                const bool owns_latent_state =
                    !std::holds_alternative<SharedKvConsumer>(layout.kv_sharing);
                if (!owns_latent_state) continue;
                const auto& latent = *layout.latent_state();
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    throw std::invalid_argument(
                        "CUDA latent attention requires BF16 local state storage");
                }
                attention->state = LatentAttentionRuntimeState{};
                auto& latent_state = std::get<LatentAttentionRuntimeState>(attention->state);
                latent_state.latent_key_cache.reset(
                    static_cast<size_t>(max_context_) * latent.latent_rank);
                latent_state.latent_value_cache.reset(
                    static_cast<size_t>(max_context_) * latent.latent_rank);
                if (latent.decoupled_rope && latent.rope_head_dim != 0) {
                    latent_state.latent_key_rope_cache.reset(
                        static_cast<size_t>(max_context_) * latent.rope_head_dim);
                }
                continue;
            }
            if (!attention->key) continue;
            const size_t cache_elements = static_cast<size_t>(max_context_) *
                static_cast<size_t>(layout.key_value_width());
            const size_t scale_elements = static_cast<size_t>(max_context_) *
                static_cast<size_t>(layout.key_value_heads);
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->state = OrdinaryInt8KvState{};
                auto& ordinary_state = std::get<OrdinaryInt8KvState>(attention->state);
                ordinary_state.key_cache.reset(cache_elements);
                ordinary_state.value_cache.reset(cache_elements);
                ordinary_state.key_scales.reset(scale_elements);
                ordinary_state.value_scales.reset(scale_elements);
            } else {
                attention->state = OrdinaryBf16KvState{};
                auto& ordinary_state = std::get<OrdinaryBf16KvState>(attention->state);
                ordinary_state.key_cache.reset(cache_elements);
                ordinary_state.value_cache.reset(cache_elements);
            }
        }
        for (Layer& layer : resources_.mtp_.layers) {
            AttentionLayer* attention = as_attention(layer);
            if (!attention || !attention->key) continue;
            const AttentionSpec& layout = attention->layout;
            const size_t cache_elements = static_cast<size_t>(max_context_) *
                static_cast<size_t>(layout.key_value_width());
            const size_t scale_elements = static_cast<size_t>(max_context_) *
                static_cast<size_t>(layout.key_value_heads);
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                attention->state = OrdinaryInt8KvState{};
                auto& ordinary_state = std::get<OrdinaryInt8KvState>(attention->state);
                ordinary_state.key_cache.reset(cache_elements);
                ordinary_state.value_cache.reset(cache_elements);
                ordinary_state.key_scales.reset(scale_elements);
                ordinary_state.value_scales.reset(scale_elements);
            } else {
                attention->state = OrdinaryBf16KvState{};
                auto& ordinary_state = std::get<OrdinaryBf16KvState>(attention->state);
                ordinary_state.key_cache.reset(cache_elements);
                ordinary_state.value_cache.reset(cache_elements);
            }
        }
        local_kv_cache_available_ = true;
    }
    session_.position_ = 0;
    session_.next_rope_position_ = {0, 0, 0};
    session_.phase_ = SessionPhase::Empty;
    const int32_t zero = 0;
    const std::array<int32_t, 3> zero_rope{0, 0, 0};
    uint64_t seed = session_.generation_.seed;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &zero, sizeof(zero),
                             cudaMemcpyHostToDevice, stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), zero_rope.data(),
                               sizeof(zero_rope), cudaMemcpyHostToDevice, stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(sampling_.rng_state.data(), &seed, sizeof(seed),
                             cudaMemcpyHostToDevice, stream_.get()));
    sampling_.seen_tokens.zero_async(stream_.get());
    for (Layer& layer : resources_.layers_) {
        visit_layer(layer,
          [](AttentionLayer*) {},
          [&](ConvolutionLayer* convolution) {
            convolution->conv_state.zero_async(stream_.get());
          },
          [&](GatedDeltaNetLayer* gated_delta) {
            gated_delta->conv_state.zero_async(stream_.get());
            gated_delta->recurrent_state.zero_async(stream_.get());
          },
          [&](Mamba2Layer* mamba) {
            mamba->conv_state.zero_async(stream_.get());
            mamba->ssm_state.zero_async(stream_.get());
          },
          [](MlpOnlyLayer*) {});
    }
    CELEG_CUDA(cudaEventRecord(workspace_.ffn_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(workspace_.router_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaEventRecord(workspace_.prefetch_done_event_.get(), stream_.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
}

void CudaCompiledModel::allocate_prefill_workspace(int rows) {
    const CudaWorkspacePlan plan = CudaWorkspacePlan::from_program(resources_.program_);
    const size_t r = static_cast<size_t>(rows);
    workspace_.prefill_tokens_.reserve(r);
    workspace_.prefill_hidden_.reserve(r * resources_.program_.hidden);
    workspace_.prefill_residual_.reserve(r * resources_.program_.hidden);
    workspace_.prefill_normed_.reserve(r * resources_.program_.hidden);
    workspace_.prefill_op_output_.reserve(r * static_cast<size_t>(std::max(
        resources_.program_.hidden,
        static_cast<int>(plan.attention_output))));
    workspace_.prefill_qkv_.reserve(r * plan.attention_projection);
    const size_t projection_width = plan.attention_projection;
    workspace_.prefill_q_.reserve(r * projection_width);
    workspace_.prefill_latent_query_content_.reserve(r * plan.latent_query_content);
    workspace_.prefill_latent_query_rope_.reserve(r * plan.latent_query_rope);
    workspace_.prefill_latent_key_.reserve(r * plan.latent_rank);
    workspace_.prefill_latent_value_.reserve(r * plan.latent_rank);
    workspace_.prefill_latent_key_rope_.reserve(r * plan.latent_rope);
    workspace_.prefill_latent_projection_.reserve(r * plan.latent_projection);
    workspace_.prefill_latent_decompressed_.reserve(r * plan.latent_decompressed);
    workspace_.prefill_attention_gate_.reserve(r * plan.attention_output);
    workspace_.prefill_k_.reserve(r * projection_width);
    workspace_.prefill_v_.reserve(r * projection_width);
    workspace_.prefill_conv_projected_.reserve(r * 3 * resources_.program_.hidden);
    workspace_.prefill_gated_delta_qkv_.reserve(r * plan.gated_delta_qkv);
    workspace_.prefill_gated_delta_z_.reserve(r * plan.gated_delta_z);
    workspace_.prefill_gated_delta_b_.reserve(r * plan.gated_delta_b);
    workspace_.prefill_gated_delta_a_.reserve(r * plan.gated_delta_a);
    workspace_.prefill_gated_delta_output_.reserve(r * plan.gated_delta_output);
    workspace_.prefill_mamba_projected_.reserve(r * plan.mamba_projection);
    workspace_.prefill_mamba_inner_.reserve(r * plan.mamba_inner);
    workspace_.prefill_gate_up_.reserve(r * 2 * plan.ffn_intermediate);
    workspace_.prefill_activated_.reserve(r * plan.ffn_intermediate);
    workspace_.prefill_mlp_output_.reserve(r * resources_.program_.hidden);
    workspace_.moe_pf_shared_gate_.reserve(r);
    if (resources_.program_.per_layer_input.enabled) {
        const size_t packed = resources_.program_.per_layer_input.checked_elements(r);
        workspace_.prefill_per_layer_token_.reserve(packed);
        workspace_.prefill_per_layer_context_.reserve(packed);
        const size_t gate_elements =
            static_cast<size_t>(r) * static_cast<size_t>(
                resources_.program_.per_layer_input.input_size);
        workspace_.prefill_per_layer_gate_.reserve(gate_elements);
    }

    if (rows <= kMaxGemmAttentionRows) {
        const size_t scores_elems =
            static_cast<size_t>(plan.attention_query_heads) * r * r;
        workspace_.prefill_attn_scores_.reserve(scores_elems);
        workspace_.prefill_attn_probs_.reserve(scores_elems);
    } else {
        const size_t chunks = (r + kPrefillAttnChunkTokens - 1) / kPrefillAttnChunkTokens;
        const size_t partials = r * plan.attention_query_heads * chunks;
        workspace_.prefill_attn_partial_max_.reserve(partials);
        workspace_.prefill_attn_partial_denom_.reserve(partials);
        workspace_.prefill_attn_partial_accum_.reserve(partials * plan.attention_head_dim);
    }
}

void CudaCompiledModel::release_prefill_workspace() {
}

}
