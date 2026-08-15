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
    Layer& layer, LayerCommon& common_layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer_index));
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");
            const AttentionSpec& layout = attention->layout;
            const float qk_norm_epsilon = layout.query_norm.enabled()
                ? layout.query_norm.epsilon : layout.key_norm.epsilon;
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
                           resources_.program_.hidden);
                    launch_rmsnorm(workspace_.latent_projection_.data(), attention->latent_query_norm,
                                   workspace_.latent_projection_.data(), 1, latent.query_rank,
                                   latent.query_latent_norm.epsilon, stream_.get());
                    linear(workspace_.latent_projection_.data(), *attention->latent_query_expansion,
                           workspace_.qkv_output_.data(), 1,
                           layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
                           latent.query_rank);
                    launch_factorized_latent_query({
                        .query_projection = workspace_.qkv_output_.data(),
                        .expansion = attention->latent_expansion->bf16,
                        .query_content = workspace_.latent_query_content_.data(),
                        .rows = 1,
                        .query_heads = layout.query_heads,
                        .query_nope = latent.nope_head_dim,
                        .query_rope_dim = latent.rope_head_dim,
                        .latent_rank = latent.latent_rank,
                        .stream = stream_.get()});
                    launch_factorized_latent_rope({
                        .query_projection = workspace_.qkv_output_.data(),
                        .query_rope = workspace_.latent_query_rope_.data(),
                        .rows = 1,
                        .query_heads = layout.query_heads,
                        .query_nope = latent.nope_head_dim,
                        .query_rope_dim = latent.rope_head_dim,
                        .stream = stream_.get()});
                    linear(workspace_.normed_.data(), *attention->latent_key_projection,
                           workspace_.qkv_output_.data(), 1,
                           latent.latent_rank + latent.rope_head_dim, resources_.program_.hidden);
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
                        1.0f, qk_norm_epsilon, false,
                        layout.rope_position()->pairing,
                        lower_cuda_rope_scaling(*layout.rope_position()), stream_.get());
                    launch_store_latent_device(
                        workspace_.latent_key_.data(), workspace_.latent_value_.data(),
                        workspace_.latent_key_rope_.data(), owner->latent_key_cache.data(),
                        owner->latent_value_cache.data(), owner->latent_key_rope_cache.data(),
                        position_device_.data(), latent.latent_rank, latent.rope_head_dim,
                        stream_.get());
                    launch_latent_attention_device({
                        .query = {.content = workspace_.latent_query_content_.data(),
                                  .rope = workspace_.latent_query_rope_.data()},
                        .kv = {.keys = owner->latent_key_cache.data(),
                               .values = owner->latent_value_cache.data(),
                               .key_rope = owner->latent_key_rope_cache.data()},
                        .out = workspace_.op_output_.data(),
                        .extent = {.position = position_device_.data()},
                        .alibi_slopes = attention->alibi_slopes.data(),
                        .geometry = {.query_heads = layout.query_heads,
                                     .latent_rank = latent.latent_rank,
                                     .rotary_width = latent.rope_head_dim,
                                     .score_scale = layout.query_scale,
                                     .sliding_window = layout.sliding_window_size()},
                        .stream = stream_.get()});
                    launch_factorized_latent_value({
                        .latent_output = workspace_.op_output_.data(),
                        .expansion = attention->latent_expansion->bf16,
                        .value_output = workspace_.latent_decompressed_.data(),
                        .rows = 1,
                        .query_heads = layout.query_heads,
                        .query_nope = latent.nope_head_dim,
                        .value_dim = latent.value_head_dim,
                        .latent_rank = latent.latent_rank,
                        .stream = stream_.get()});
                    linear(workspace_.normed_.data(), *attention->gate,
                           workspace_.attention_gate_.data(), 1, layout.output_gate_width(),
                           resources_.program_.hidden);
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
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           layout.latent_output_width(),
                           resources_.options_.fused_residuals && !common_layer.post_attention_norm ? 1.0f : 0.0f);
                    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                                 semantics.residual.multiplier, stream_.get());
                    return;
                }
                if (layout.output_gate.enabled() || layout.multi_axis_position()) {
                    throw std::invalid_argument(
                        "CUDA latent attention does not support query gates or M-RoPE yet");
                }
                decode_phase_profile().begin(stream_.get());
                linear(workspace_.normed_.data(), *attention->latent_query,
                       workspace_.latent_query_content_.data(), 1,
                       layout.latent_query_content_width(), resources_.program_.hidden);
                if (layout.latent_query_rope_width() != 0) {
                    linear(workspace_.normed_.data(), *attention->latent_query_rope,
                           workspace_.latent_query_rope_.data(), 1,
                           layout.latent_query_rope_width(), resources_.program_.hidden);
                }
                if (attention->latent_key && attention->latent_value) {
                    linear(workspace_.normed_.data(), *attention->latent_key,
                           workspace_.latent_key_.data(), 1, latent.latent_rank,
                           resources_.program_.hidden);
                    linear(workspace_.normed_.data(), *attention->latent_value,
                           workspace_.latent_value_.data(), 1, latent.latent_rank,
                           resources_.program_.hidden);
                    if (attention->latent_key_rope && latent.decoupled_rope &&
                        latent.rope_head_dim != 0) {
                        linear(workspace_.normed_.data(), *attention->latent_key_rope,
                               workspace_.latent_key_rope_.data(), 1,
                               latent.rope_head_dim, resources_.program_.hidden);
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
                        qk_norm_epsilon, false,
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
                const float score_scale = layout.query_scale;
                launch_latent_attention_device({
                    .query = {.content = workspace_.latent_query_content_.data(),
                              .rope = layout.latent_query_rope_width() != 0
                                          ? workspace_.latent_query_rope_.data()
                                          : nullptr},
                    .kv = {.keys = owner->latent_key_cache.data(),
                           .values = owner->latent_value_cache.data(),
                           .key_rope = owner->latent_key_rope_cache.data()},
                    .out = workspace_.op_output_.data(),
                    .extent = {.position = position_device_.data()},
                    .alibi_slopes = attention->alibi_slopes.data(),
                    .geometry = {.query_heads = layout.query_heads,
                                 .latent_rank = latent.latent_rank,
                                 .rotary_width = latent.decoupled_rope
                                                     ? latent.rope_head_dim : 0,
                                 .score_scale = score_scale,
                                 .sliding_window = layout.sliding_window_size()},
                    .stream = stream_.get()});
                decode_phase_profile().end(DecodePhase::Attention, stream_.get());
                decode_phase_profile().begin(stream_.get());
                linear(workspace_.op_output_.data(), *attention->out,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       layout.latent_query_content_width(),
                       resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                           std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f);
                launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                             semantics.residual.multiplier,
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
                workspace_.normed_.data(), 1, resources_.program_.hidden);
            linear(workspace_.normed_.data(), *attention->query, q,
                   1, query_projection_width, resources_.program_.hidden);
            if (attention->key && attention->value) {
                linear(workspace_.normed_.data(), *attention->key, k,
                       1, layout.key_value_width(), resources_.program_.hidden);
                linear(workspace_.normed_.data(), *attention->value, v,
                       1, layout.key_value_width(), resources_.program_.hidden);
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
                        qk_norm_epsilon,
                        layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
                } else {
                    launch_dynamic_qk_norm_rope_device(
                        q, attention->key ? k : nullptr, attention->q_norm, attention->k_norm,
                        layout.query_heads, layout.key_value_heads, layout.head_dim,
                        position_device_.data(), static_cast<float>(rope->theta),
                        static_cast<float>(rope->rotary_fraction), qk_norm_epsilon,
                        layout.has_query_key_norm(), lower_cuda_rope_scaling(*rope), stream_.get());
                }
            }
            launch_scale(q, layout.query_width(), layout.query_scale, stream_.get());
            decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
            decode_phase_profile().begin(stream_.get());
            AttentionRequest attention_request;
            attention_request.kv_format = resources_.options_.kv_cache_mode;
            attention_request.operation = AttentionOperation::Decode;
            attention_request.layout = AttentionKvLayout::Contiguous;
            attention_request.position_source = AttentionPositionSource::DeviceCounter;
            attention_request.bias = attention->alibi_slopes.data()
                ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
            attention_request.fast_attention = resources_.options_.fast_attention;
            attention_request.segmented_attention = session_.active_segmented_attention_;
            attention_request.head_dim = owner_layout.head_dim;
            const AttentionCapability attention_plan =
                require_attention_capability(attention_request);
            const bool int8_kv = attention_request.kv_format == KvCacheMode::Int8;
            if (attention->key && attention->value) {
                if (int8_kv) {
                    launch_store_kv_int8_device(
                        k, v, owner->key_cache_int8.data(), owner->value_cache_int8.data(),
                        owner->key_cache_scales.data(), owner->value_cache_scales.data(),
                        position_device_.data(), owner_layout.key_value_heads,
                        owner_layout.head_dim, stream_.get());
                } else {
                    launch_store_kv_device(
                        k, v, owner->key_cache.data(), owner->value_cache.data(),
                        position_device_.data(), owner_layout.key_value_width(), stream_.get());
                }
            }
            const GqaGeometry attention_geometry{
                .q_heads = layout.query_heads,
                .kv_heads = owner_layout.key_value_heads,
                .head_dim = owner_layout.head_dim,
                .sliding_window = layout.sliding_window_size()};
            const AttentionExtent attention_extent{.position = position_device_.data()};
            const Bf16KvView bf16_kv{.keys = owner->key_cache.data(),
                                     .values = owner->value_cache.data()};
            const Int8KvView int8_kv_view{
                .keys = owner->key_cache_int8.data(),
                .values = owner->value_cache_int8.data(),
                .key_scales = owner->key_cache_scales.data(),
                .value_scales = owner->value_cache_scales.data()};
            const AttentionSegmentation attention_segmentation{
                .chunk_tokens = resources_.options_.attention_chunk_tokens,
                .chunks = workspace_.attention_chunks_,
                .partial_max = workspace_.attention_partial_max_.data(),
                .partial_denom = workspace_.attention_partial_denom_.data(),
                .partial_accum = workspace_.attention_partial_accum_.data()};
            switch (attention_plan.algorithm) {
            case AttentionAlgorithm::Alibi:
                if (int8_kv) {
                    launch_gqa_decode_alibi_int8_device({
                        .query = q,
                        .kv = int8_kv_view,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .alibi_slopes = attention->alibi_slopes.data(),
                        .stream = stream_.get()});
                } else {
                    launch_gqa_decode_alibi_device({
                        .query = q,
                        .kv = bf16_kv,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .alibi_slopes = attention->alibi_slopes.data(),
                        .stream = stream_.get()});
                }
                break;
            case AttentionAlgorithm::Segmented:
                if (int8_kv) {
                    launch_gqa_decode_segmented_int8_device({
                        .query = q,
                        .kv = int8_kv_view,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .segmentation = attention_segmentation,
                        .stream = stream_.get()});
                } else {
                    launch_gqa_decode_segmented_device({
                        .query = q,
                        .kv = bf16_kv,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .segmentation = attention_segmentation,
                        .stream = stream_.get()});
                }
                break;
            case AttentionAlgorithm::Online:
                if (int8_kv) {
                    launch_gqa_decode_online_int8_device({
                        .query = q,
                        .kv = int8_kv_view,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .stream = stream_.get()});
                } else {
                    launch_gqa_decode_online_device({
                        .query = q,
                        .kv = bf16_kv,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .stream = stream_.get()});
                }
                break;
            case AttentionAlgorithm::Strict:
                if (int8_kv) {
                    launch_gqa_decode_strict_int8_device({
                        .query = q,
                        .kv = int8_kv_view,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .stream = stream_.get()});
                } else {
                    launch_gqa_decode_strict_device({
                        .query = q,
                        .kv = bf16_kv,
                        .out = workspace_.op_output_.data(),
                        .geometry = attention_geometry,
                        .extent = attention_extent,
                        .stream = stream_.get()});
                }
                break;
            case AttentionAlgorithm::Flash:
            case AttentionAlgorithm::Gemm:
                throw UnsupportedAttentionCapability(attention_plan);
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
                           layout.query_width(), resources_.program_.hidden);
                    gate = workspace_.attention_gate_.data();
                }
                launch_sigmoid_multiply(workspace_.op_output_.data(),
                                        gate,
                                        layout.query_width(), stream_.get());
            }
            decode_phase_profile().end(DecodePhase::Attention, stream_.get());
            decode_phase_profile().begin(stream_.get());
            linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
                   1, resources_.program_.hidden, layout.query_width(),
                   resources_.options_.fused_residuals && !common_layer.post_attention_norm &&
                       std::holds_alternative<std::monostate>(semantics.feed_forward) ? 0.0f : 1.0f);
            launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                         semantics.residual.multiplier,
                         stream_.get());
            decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
            }
}

}
