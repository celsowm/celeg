#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "attention_output_gate.hpp"
#include "attention_qk_prepare.hpp"
#include "backend/cuda/phase_profile.hpp"
#include "kernels/kernels.cuh"
#include "kernels/attention_output.hpp"
#include "backend/cuda/paged_kv.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_standard_attention(
    Layer& layer, int layer_index) {
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
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        *attention, layer_index, resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    const AttentionSpec& owner_layout = owner.layout;
    if (execution.kind != AttentionExecutionKind::Standard) {
        throw std::logic_error("standard CUDA attention has a non-standard descriptor");
    }

    decode_phase_profile().begin(stream_.get());
    const CudaQkvProjectionView qkv = project_standard_attention_qkv(*attention);
    __nv_bfloat16* q = qkv.query;
    __nv_bfloat16* k = qkv.key;
    __nv_bfloat16* v = qkv.value;
    decode_phase_profile().end(DecodePhase::Projection, stream_.get());

    decode_phase_profile().begin(stream_.get());
    prepare_cuda_attention_qk({
        .layout = &layout,
        .query = q,
        .key = attention->key ? k : nullptr,
        .query_norm = attention->q_norm,
        .key_norm = attention->k_norm,
        .norm_epsilon = qk_norm_epsilon,
        .position_mode = CudaQkPositionMode::DeviceScalar,
        .device_position = position_device_.data(),
        .stream = stream_.get()});
    decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());

    decode_phase_profile().begin(stream_.get());
    const CudaDecodeAttentionPolicy attention_policy = plan_cuda_decode_attention(
        layout,
        resources_.options().kv_cache_mode,
        AttentionKvLayout::Contiguous,
        AttentionPositionSource::DeviceCounter,
        resources_.options().fast_attention,
        session_.active_segmented_attention_,
        owner_layout.head_dim);
    store_standard_attention_kv_contiguous(
        *attention, owner, attention_policy.plan, k, v);

    dispatch_cuda_contiguous_decode_attention({
        .plan = attention_policy.plan,
        .position_mode = CudaDecodePositionMode::DeviceCounter,
        .block_sparse = attention_policy.block_sparse,
        .query = q,
        .bf16_kv = cuda_bf16_kv_view(owner),
        .int8_kv = cuda_int8_kv_view(owner),
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
        .relative_bias = cuda_relative_position_bias_view(*attention),
        .stream = stream_.get()});

    if (const auto* transform = std::get_if<OrthogonalizeCurrentValueSpec>(
            &layout.output_transform)) {
        launch_orthogonalize_current_value(
            workspace_.op_output_.data(), v, 1, layout.query_heads,
            layout.key_value_heads, layout.head_dim,
            transform->minimum_norm_squared, stream_.get());
    }
    apply_cuda_graph_attention_gate(*this, *attention, q);
    decode_phase_profile().end(DecodePhase::Attention, stream_.get());

    decode_phase_profile().begin(stream_.get());
    project_standard_attention_output(*attention, semantics);
    decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
}

}
