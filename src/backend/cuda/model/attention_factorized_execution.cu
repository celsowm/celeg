#include "detail/compiled_model.hpp"
#include "attention_kv_store.hpp"
#include "attention_latent_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "attention_qk_prepare.hpp"
#include "residual_fusion.hpp"
#include "backend/cuda/phase_profile.hpp"
#include "kernels/kernels.cuh"
#include "kernels/attention_output.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_factorized_latent_attention(
    Layer& layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<std::size_t>(layer_index));
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention || compiled_attention->execution.kind !=
        AttentionExecutionKind::FactorizedLatent) {
        throw std::logic_error(
            "CUDA factorized attention has an invalid execution descriptor");
    }
    const CompiledAttentionExecution& execution = compiled_attention->execution;
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");
    if (resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 state storage");
    }
    const AttentionSpec& layout = attention->layout;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;
    const bool fuse_mixer_residual = cuda_can_fuse_mixer_residual(
        resources_.options().fused_residuals, semantics);
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        *attention, layer_index, resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    const __nv_bfloat16* latent_expansion =
        require_cuda_factorized_latent_bindings(*attention);
    const auto& latent = *layout.latent_state();
    const auto* factorized = latent.factorized_projection();

    decode_phase_profile().begin(stream_.get());
    project_cuda_factorized_latent_attention_qkv(*this, *attention);
    decode_phase_profile().end(DecodePhase::Projection, stream_.get());

    prepare_cuda_factorized_latent_attention_qk({
        .layout = &layout,
        .query_rope = workspace_.latent_query_rope_.data(),
        .key_rope = workspace_.latent_key_rope_.data(),
        .norm_epsilon = qk_norm_epsilon,
        .rows = 1,
        .device_position = position_device_.data(),
        .stream = stream_.get()});
    store_cuda_latent_kv_contiguous(*this, *attention, owner);
    dispatch_cuda_latent_attention_contiguous(*this, *attention, owner);
    launch_factorized_latent_value({
        .latent_output = workspace_.op_output_.data(),
        .expansion = latent_expansion,
        .value_output = workspace_.latent_decompressed_.data(),
        .rows = 1,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .value_dim = factorized->value_head_dim,
        .latent_rank = latent.latent_rank,
        .stream = stream_.get()});
    linear(workspace_.normed_.data(), *attention->gate,
           workspace_.attention_gate_.data(), 1, layout.output_gate_width(),
           resources_.program_.hidden);
    if (execution.gate_granularity == AttentionGateGranularity::HeadWise) {
        launch_sigmoid_multiply_headwise(workspace_.latent_decompressed_.data(),
            workspace_.attention_gate_.data(), 1, layout.query_heads,
            factorized->value_head_dim, stream_.get());
    } else {
        launch_sigmoid_multiply(workspace_.latent_decompressed_.data(),
            workspace_.attention_gate_.data(), layout.latent_output_width(),
            stream_.get());
    }
    linear(workspace_.latent_decompressed_.data(), *attention->out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           layout.latent_output_width(), fuse_mixer_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

}
