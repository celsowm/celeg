#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "backend/cuda/phase_profile.hpp"
#include "kernels/kernels.cuh"
#include "kernels/attention_output.hpp"
#include "kernels/rope_pairing.hpp"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_layout.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_standard_attention(
    Layer& layer, LayerCommon& common_layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<size_t>(layer_index));
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention) {
        throw std::logic_error("CUDA attention has no compiled attention program");
    }
    const CompiledAttentionExecution& execution = compiled_attention->execution;
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");

    const AttentionSpec& layout = attention->layout;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;
    const bool fuse_mixer_residual = resources_.options().fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    AttentionLayer* owner = attention;
    if (attention->kv_owner_layer >= 0) {
        owner = as_attention(resources_.layers_.at(
            static_cast<size_t>(attention->kv_owner_layer)));
        if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
    }
    const AttentionSpec& owner_layout = owner->layout;
    if (execution.kind != AttentionExecutionKind::Standard) {
        throw std::logic_error("standard CUDA attention has a non-standard descriptor");
    }

    __nv_bfloat16* q = workspace_.qkv_output_.data();
    const int query_projection_width = attention->query->rows;
    const bool output_gate = layout.output_gate.has_value();
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
    if (layout.has_query_key_norm()) {
        launch_attention_qk_norm(
            layout, q, attention->key ? k : nullptr,
            attention->q_norm, attention->k_norm, 1, stream_.get());
    }
    if (const auto* rope = layout.rope_position()) {
        if (rope->pairing == RopePairingKind::AdjacentPairs) {
            launch_adjacent_qk_norm_rope_positions(
                q, attention->key ? k : nullptr,
                nullptr, nullptr, 1,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                position_device_.data(), static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction),
                qk_norm_epsilon, false, lower_cuda_rope_scaling(*rope), stream_.get());
        } else {
            launch_dynamic_qk_norm_rope_device(
                q, attention->key ? k : nullptr, nullptr, nullptr,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                position_device_.data(), static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction), qk_norm_epsilon,
                false, lower_cuda_rope_scaling(*rope), rope->pairing, stream_.get());
        }
    }
    launch_scale(q, layout.query_width(), cuda_query_prescale(layout), stream_.get());
    decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());

    decode_phase_profile().begin(stream_.get());
    const auto* block_sparse = std::get_if<BlockSparsePattern>(&layout.pattern);
    AttentionRequest attention_request;
    attention_request.kv_format = resources_.options().kv_cache_mode;
    attention_request.operation = AttentionOperation::Decode;
    attention_request.layout = AttentionKvLayout::Contiguous;
    attention_request.position_source = AttentionPositionSource::DeviceCounter;
    attention_request.bias = attention->alibi_slopes.data()
        ? AttentionPositionBias::Alibi : AttentionPositionBias::None;
    attention_request.fast_attention =
        resources_.options().fast_attention && block_sparse == nullptr;
    attention_request.segmented_attention =
        session_.active_segmented_attention_ && block_sparse == nullptr;
    attention_request.head_dim = owner_layout.head_dim;
    const AttentionCapability attention_plan =
        require_attention_capability(attention_request);
    const bool int8_kv = attention_request.kv_format == KvCacheMode::Int8;

    if (attention->key && attention->value) {
        if (int8_kv) {
            launch_store_kv_int8_device(
                k, v, owner->key_cache_int8_ptr(), owner->value_cache_int8_ptr(),
                owner->key_cache_scales_ptr(), owner->value_cache_scales_ptr(),
                position_device_.data(), owner_layout.key_value_heads,
                owner_layout.head_dim, stream_.get());
        } else {
            launch_store_kv_device(
                k, v, owner->key_cache_bf16(), owner->value_cache_bf16(),
                position_device_.data(), owner_layout.key_value_width(), stream_.get());
        }
    }

    dispatch_cuda_contiguous_decode_attention({
        .plan = attention_plan,
        .position_mode = CudaDecodePositionMode::DeviceCounter,
        .block_sparse = block_sparse,
        .query = q,
        .bf16_kv = {.keys = owner->key_cache_bf16(),
                    .values = owner->value_cache_bf16()},
        .int8_kv = {.keys = owner->key_cache_int8_ptr(),
                    .values = owner->value_cache_int8_ptr(),
                    .key_scales = owner->key_cache_scales_ptr(),
                    .value_scales = owner->value_cache_scales_ptr()},
        .out = workspace_.op_output_.data(),
        .geometry = make_cuda_gqa_geometry(layout, owner_layout),
        .extent = {.position = position_device_.data()},
        .segmentation = {
            .segments = workspace_.attention_segments_,
            .min_segments = workspace_.attention_min_segments_,
            .partial_max = workspace_.attention_partial_max_.data(),
            .partial_denom = workspace_.attention_partial_denom_.data(),
            .partial_accum = workspace_.attention_partial_accum_.data()},
        .alibi_slopes = attention->alibi_slopes.data(),
        .stream = stream_.get()});

    if (const auto* transform = std::get_if<OrthogonalizeCurrentValueSpec>(
            &layout.output_transform)) {
        launch_orthogonalize_current_value(
            workspace_.op_output_.data(), v, 1, layout.query_heads,
            layout.key_value_heads, layout.head_dim,
            transform->minimum_norm_squared, stream_.get());
    }
    if (output_gate) {
        const __nv_bfloat16* gate = q + layout.query_width();
        if (!layout.output_gate->packed_with_query) {
            linear(workspace_.normed_.data(), *attention->gate,
                   workspace_.attention_gate_.data(), 1,
                   layout.query_width(), resources_.program_.hidden);
            gate = workspace_.attention_gate_.data();
        }
        launch_sigmoid_multiply(workspace_.op_output_.data(), gate,
                                layout.query_width(), stream_.get());
    }
    decode_phase_profile().end(DecodePhase::Attention, stream_.get());

    decode_phase_profile().begin(stream_.get());
    linear(workspace_.op_output_.data(), *attention->out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, layout.query_width(),
           fuse_mixer_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
    decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
}

}
