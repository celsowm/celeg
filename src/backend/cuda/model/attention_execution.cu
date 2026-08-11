#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/kernels/attention_output.hpp"
#include "celeg/backend/cuda/kernels/rope_pairing.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_attention(
    Layer& layer, LayerCommon& common_layer) {
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");
            const AttentionSpec& layout = attention->layout;
            AttentionLayer* owner = attention;
            if (attention->kv_owner_layer >= 0) {
                owner = as_attention(resources_.layers_.at(
                    static_cast<size_t>(attention->kv_owner_layer)));
                if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
            }
            const AttentionSpec& owner_layout = owner->layout;
            if (layout.uses_latent_state()) {
                if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                    throw std::invalid_argument(
                        "CUDA latent attention requires BF16 state storage");
                }
                const auto& latent = *layout.latent_state();
                if (latent.factorized) {
                    decode_phase_profile().begin(stream_.get());
                    linear(workspace_.normed_.data(), *attention->latent_query_projection,
                           workspace_.latent_projection_.data(), 1, latent.query_rank,
                           resources_.shape_.hidden);
                    launch_rmsnorm(workspace_.latent_projection_.data(), attention->latent_query_norm,
                                   workspace_.latent_projection_.data(), 1, latent.query_rank,
                                   latent.query_latent_norm.epsilon, stream_.get());
                    linear(workspace_.latent_projection_.data(), *attention->latent_query_expansion,
                           workspace_.qkv_output_.data(), 1,
                           layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                           latent.query_rank);
                    launch_factorized_latent_query(
                        workspace_.qkv_output_.data(), attention->latent_expansion->bf16,
                        workspace_.latent_query_content_.data(), 1, layout.query_heads,
                        latent.nope_head_dim, latent.rope_head_dim, latent.latent_rank,
                        stream_.get());
                    launch_factorized_latent_rope(
                        workspace_.qkv_output_.data(), workspace_.latent_query_rope_.data(),
                        1, layout.query_heads, latent.nope_head_dim, latent.rope_head_dim,
                        stream_.get());
                    linear(workspace_.normed_.data(), *attention->latent_key_projection,
                           workspace_.qkv_output_.data(), 1,
                           latent.latent_rank + latent.rope_head_dim, resources_.shape_.hidden);
                    launch_rmsnorm(workspace_.qkv_output_.data(), attention->latent_key_norm,
                                   workspace_.latent_key_.data(), 1, latent.latent_rank,
                                   latent.key_latent_norm.epsilon, stream_.get());
                    CELEG_CUDA(cudaMemcpyAsync(workspace_.latent_value_.data(),
                        workspace_.latent_key_.data(),
                        static_cast<size_t>(latent.latent_rank) * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_.get()));
                    CELEG_CUDA(cudaMemcpyAsync(workspace_.latent_key_rope_.data(),
                        workspace_.qkv_output_.data() + latent.latent_rank,
                        static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream_.get()));
                    decode_phase_profile().end(DecodePhase::Projection, stream_.get());
                    launch_qk_norm_rope_positions(
                        workspace_.latent_query_rope_.data(), workspace_.latent_key_rope_.data(),
                        nullptr, nullptr, 1, layout.query_heads, 1, latent.rope_head_dim,
                        position_device_.data(), static_cast<float>(layout.rope_position()->theta),
                        1.0f, resources_.shape_.numerical_policy.norm_eps, false,
                        layout.rope_position()->pairing,
                        lower_cuda_rope_scaling(*layout.rope_position()), stream_.get());
                    launch_store_latent_device(
                        workspace_.latent_key_.data(), workspace_.latent_value_.data(),
                        workspace_.latent_key_rope_.data(), owner->latent_key_cache.data(),
                        owner->latent_value_cache.data(), owner->latent_key_rope_cache.data(),
                        position_device_.data(), latent.latent_rank, latent.rope_head_dim,
                        stream_.get());
                    launch_latent_attention_device(
                        workspace_.latent_query_content_.data(), workspace_.latent_query_rope_.data(),
                        owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                        owner->latent_key_rope_cache.data(), workspace_.op_output_.data(),
                        position_device_.data(), attention->alibi_slopes.data(), layout.query_heads,
                        latent.latent_rank, latent.rope_head_dim,
                        layout.query_scale * resources_.shape_.numerical_policy.attention_multiplier,
                        layout.sliding_window_size(), stream_.get());
                    launch_factorized_latent_value(
                        workspace_.op_output_.data(), attention->latent_expansion->bf16,
                        workspace_.latent_decompressed_.data(), 1, layout.query_heads,
                        latent.nope_head_dim, latent.value_head_dim, latent.latent_rank,
                        stream_.get());
                    linear(workspace_.normed_.data(), *attention->gate,
                           workspace_.attention_gate_.data(), 1, layout.output_gate_width(),
                           resources_.shape_.hidden);
                    if (layout.output_gate.granularity == AttentionGateGranularity::HeadWise) {
                        launch_sigmoid_multiply_headwise(workspace_.latent_decompressed_.data(),
                            workspace_.attention_gate_.data(), 1, layout.query_heads,
                            latent.value_head_dim, stream_.get());
                    } else {
                        launch_sigmoid_multiply(workspace_.latent_decompressed_.data(),
                            workspace_.attention_gate_.data(), layout.latent_output_width(),
                            stream_.get());
                    }
                    linear(workspace_.latent_decompressed_.data(), *attention->out,
                           workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                           layout.latent_output_width(),
                           resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
                    launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                                 resources_.shape_.numerical_policy.residual_multiplier, stream_.get());
                    return;
                }
                if (layout.output_gate.enabled() || layout.multi_axis_position()) {
                    throw std::invalid_argument(
                        "CUDA latent attention does not support query gates or M-RoPE yet");
                }
                decode_phase_profile().begin(stream_.get());
                linear(workspace_.normed_.data(), *attention->latent_query,
                       workspace_.latent_query_content_.data(), 1,
                       layout.latent_query_content_width(), resources_.shape_.hidden);
                if (layout.latent_query_rope_width() != 0) {
                    linear(workspace_.normed_.data(), *attention->latent_query_rope,
                           workspace_.latent_query_rope_.data(), 1,
                           layout.latent_query_rope_width(), resources_.shape_.hidden);
                }
                if (attention->latent_key && attention->latent_value) {
                    linear(workspace_.normed_.data(), *attention->latent_key,
                           workspace_.latent_key_.data(), 1, latent.latent_rank,
                           resources_.shape_.hidden);
                    linear(workspace_.normed_.data(), *attention->latent_value,
                           workspace_.latent_value_.data(), 1, latent.latent_rank,
                           resources_.shape_.hidden);
                    if (attention->latent_key_rope && latent.decoupled_rope &&
                        latent.rope_head_dim != 0) {
                        linear(workspace_.normed_.data(), *attention->latent_key_rope,
                               workspace_.latent_key_rope_.data(), 1,
                               latent.rope_head_dim, resources_.shape_.hidden);
                    }
                }
                decode_phase_profile().end(DecodePhase::Projection, stream_.get());
                decode_phase_profile().begin(stream_.get());
                if (const auto* rope = layout.rope_position();
                    rope && attention->latent_key_rope && latent.decoupled_rope &&
                    latent.rope_head_dim != 0) {
                    if (rope->pairing != RopePairingKind::SplitHalf) {
                        throw std::invalid_argument(
                            "CUDA latent attention requires split-half RoPE pairing");
                    }
                    launch_dynamic_qk_norm_rope_device(
                        workspace_.latent_query_rope_.data(),
                        attention->latent_key ? workspace_.latent_key_rope_.data() : nullptr,
                        nullptr, nullptr, layout.query_heads, 1,
                        latent.rope_head_dim, position_device_.data(),
                        static_cast<float>(rope->theta), 1.0f,
                        resources_.shape_.numerical_policy.norm_eps, false,
                        lower_cuda_rope_scaling(*rope), stream_.get());
                }
                decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
                decode_phase_profile().begin(stream_.get());
                if (attention->latent_key && attention->latent_value) {
                    launch_store_latent_device(
                        workspace_.latent_key_.data(), workspace_.latent_value_.data(),
                        attention->latent_key_rope && latent.decoupled_rope &&
                        latent.rope_head_dim != 0
                            ? workspace_.latent_key_rope_.data() : nullptr,
                        owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                        owner->latent_key_rope_cache.data(), position_device_.data(),
                        latent.latent_rank,
                        latent.decoupled_rope ? latent.rope_head_dim : 0,
                        stream_.get());
                }
                const float score_scale = layout.query_scale *
                    resources_.shape_.numerical_policy.attention_multiplier;
                launch_latent_attention_device(
                    workspace_.latent_query_content_.data(),
                    layout.latent_query_rope_width() != 0
                        ? workspace_.latent_query_rope_.data() : nullptr,
                    owner->latent_key_cache.data(), owner->latent_value_cache.data(),
                    owner->latent_key_rope_cache.data(), workspace_.op_output_.data(),
                    position_device_.data(), attention->alibi_slopes.data(),
                    layout.query_heads, latent.latent_rank,
                    latent.decoupled_rope ? latent.rope_head_dim : 0,
                    score_scale, layout.sliding_window_size(), stream_.get());
                decode_phase_profile().end(DecodePhase::Attention, stream_.get());
                decode_phase_profile().begin(stream_.get());
                linear(workspace_.op_output_.data(), *attention->out,
                       workspace_.hidden_.data(), 1, resources_.shape_.hidden,
                       layout.latent_query_content_width(),
                       resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                           resources_.shape_.mamba2_layer_count == 0 ? 1.0f : 0.0f);
                launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden,
                             resources_.shape_.numerical_policy.residual_multiplier,
                             stream_.get());
                decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
            } else {
            __nv_bfloat16* q = workspace_.qkv_output_.data();
            const int query_projection_width = attention->query->rows;
            const bool output_gate = layout.output_gate.enabled();
            __nv_bfloat16* k = q + query_projection_width;
            __nv_bfloat16* v = k + layout.key_value_width();
            decode_phase_profile().begin(stream_.get());
            {
            auto native_fanout = native_fanout_scope(
                workspace_.normed_.data(), 1, resources_.shape_.hidden);
            linear(workspace_.normed_.data(), *attention->query, q,
                   1, query_projection_width, resources_.shape_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k,
                       1, layout.key_value_width(), resources_.shape_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v,
                       1, layout.key_value_width(), resources_.shape_.hidden);
            }
            }
            decode_phase_profile().end(DecodePhase::Projection, stream_.get());
            decode_phase_profile().begin(stream_.get());
            if (const auto* rope = layout.rope_position()) {
                if (rope->pairing == RopePairingKind::AdjacentPairs) {
                    launch_adjacent_qk_norm_rope_positions(
                        q, attention->key ? k : nullptr,
                        attention->q_norm, attention->k_norm, 1,
                        layout.query_heads, layout.key_value_heads, layout.head_dim,
                        position_device_.data(), static_cast<float>(rope->theta),
                        static_cast<float>(rope->rotary_fraction),
                        resources_.shape_.numerical_policy.norm_eps,
                        layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
                } else {
                    launch_dynamic_qk_norm_rope_device(
                        q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                        layout.query_heads, layout.key_value_heads, layout.head_dim,
                        position_device_.data(), static_cast<float>(rope->theta),
                        static_cast<float>(rope->rotary_fraction), resources_.shape_.numerical_policy.norm_eps,
                        layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
                }
            }
            launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());
            decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
            decode_phase_profile().begin(stream_.get());
            if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
                if (attention->key && attention->value) launch_store_kv_int8_device(
                    k, v, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                    owner->key_cache_scales.data(), owner->value_cache_scales.data(), position_device_.data(),
                    owner_layout.key_value_heads, owner_layout.head_dim, stream_.get());
                if (attention->alibi_slopes.data()) {
                    launch_gqa_decode_alibi_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        attention->alibi_slopes.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else if (session_.active_segmented_attention_) {
                    launch_gqa_decode_segmented_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        resources_.options_.attention_chunk_tokens, workspace_.attention_chunks_,
                        layout.sliding_window_size(),
                        workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else {
                    launch_gqa_decode_strict_int8_device(
                        q, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(), workspace_.op_output_.data(),
                        position_device_.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                }
            } else {
                if (attention->key && attention->value) launch_store_kv_device(
                    k, v, owner->key_cache.data(), owner->value_cache.data(),
                    position_device_.data(), owner_layout.key_value_width(), stream_.get());
                if (attention->alibi_slopes.data()) {
                    launch_gqa_decode_alibi_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        attention->alibi_slopes.data(), layout.query_heads,
                        owner_layout.key_value_heads, owner_layout.head_dim,
                        layout.sliding_window_size(), stream_.get());
                } else if (session_.active_segmented_attention_) {
                    launch_gqa_decode_segmented_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, resources_.options_.attention_chunk_tokens,
                        workspace_.attention_chunks_, layout.sliding_window_size(),
                        workspace_.attention_partial_max_.data(),
                        workspace_.attention_partial_denom_.data(),
                        workspace_.attention_partial_accum_.data(), stream_.get());
                } else if (resources_.options_.fast_attention) {
                    launch_gqa_decode_online_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window_size(), stream_.get());
                } else {
                    launch_gqa_decode_strict_device(
                        q, owner->key_cache.data(), owner->value_cache.data(),
                        workspace_.op_output_.data(), position_device_.data(),
                        layout.query_heads, owner_layout.key_value_heads,
                        owner_layout.head_dim, layout.sliding_window_size(), stream_.get());
                }
            }
            if (const auto* transform = std::get_if<OrthogonalizeCurrentValueSpec>(
                    &layout.output_transform)) {
                launch_orthogonalize_current_value(
                    workspace_.op_output_.data(), v, 1, layout.query_heads,
                    layout.key_value_heads, layout.head_dim,
                    transform->minimum_norm_squared, stream_.get());
            }
            if (output_gate) {
                const __nv_bfloat16* gate = q + layout.query_width();
                if (!layout.output_gate.packed_with_query) {
                    linear(workspace_.normed_.data(), *attention->gate,
                           workspace_.attention_gate_.data(), 1,
                           layout.query_width(), resources_.shape_.hidden);
                    gate = workspace_.attention_gate_.data();
                }
                launch_sigmoid_multiply(workspace_.op_output_.data(),
                                        gate,
                                        layout.query_width(), stream_.get());
            }
            decode_phase_profile().end(DecodePhase::Attention, stream_.get());
            decode_phase_profile().begin(stream_.get());
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.shape_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                       resources_.shape_.mamba2_layer_count == 0 ? 1.0f : 0.0f);
            launch_scale(workspace_.hidden_.data(), resources_.shape_.hidden, resources_.shape_.numerical_policy.residual_multiplier,
                         stream_.get());
            decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
            }
}

} // namespace celeg
