#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"
#include "residual_fusion.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "backend/cuda/phase_profile.hpp"
#include "kernels/kernels.cuh"
#include "kernels/attention_output.hpp"
#include "kernels/rope_pairing.hpp"
#include "backend/cuda/weight_layout.hpp"

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
    const auto& latent = *layout.latent_state();
    const auto* factorized = latent.factorized_projection();
    if (!factorized || !attention->latent_query_projection ||
        !attention->latent_query_expansion || !attention->latent_key_projection ||
        !attention->latent_key_norm || !attention->latent_query_norm ||
        !attention->latent_expansion || !attention->gate) {
        throw std::logic_error("factorized CUDA attention has incomplete bindings");
    }
    const auto* latent_expansion =
        std::get_if<Bf16LinearStorage>(&attention->latent_expansion->storage);
    if (!latent_expansion || !latent_expansion->data) {
        throw std::logic_error(
            "factorized CUDA attention requires BF16 latent expansion storage");
    }
    decode_phase_profile().begin(stream_.get());
    linear(workspace_.normed_.data(), *attention->latent_query_projection,
           workspace_.latent_projection_.data(), 1, factorized->query_rank,
           resources_.program_.hidden);
    launch_rmsnorm(workspace_.latent_projection_.data(), attention->latent_query_norm,
                   workspace_.latent_projection_.data(), 1, factorized->query_rank,
                   factorized->query_latent_norm.epsilon, stream_.get());
    linear(workspace_.latent_projection_.data(), *attention->latent_query_expansion,
           workspace_.qkv_output_.data(), 1,
           layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
           factorized->query_rank);
    launch_factorized_latent_query({
        .query_projection = workspace_.qkv_output_.data(),
        .expansion = latent_expansion->data,
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
                   factorized->key_latent_norm.epsilon, stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(workspace_.latent_value_.data(),
        workspace_.latent_key_.data(),
        static_cast<std::size_t>(latent.latent_rank) * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream_.get()));
    CELEG_CUDA(cudaMemcpyAsync(workspace_.latent_key_rope_.data(),
        workspace_.qkv_output_.data() + latent.latent_rank,
        static_cast<std::size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream_.get()));
    decode_phase_profile().end(DecodePhase::Projection, stream_.get());
    launch_qk_norm_rope_positions(
        workspace_.latent_query_rope_.data(), workspace_.latent_key_rope_.data(),
        nullptr, nullptr, 1, layout.query_heads, 1, latent.rope_head_dim,
        position_device_.data(), static_cast<float>(layout.rope_position()->theta),
        1.0f, qk_norm_epsilon, false, layout.rope_position()->pairing,
        lower_cuda_rope_scaling(*layout.rope_position()), stream_.get());
    launch_store_latent_device(
        workspace_.latent_key_.data(), workspace_.latent_value_.data(),
        workspace_.latent_key_rope_.data(), owner.latent_key_cache_ptr(),
        owner.latent_value_cache_ptr(), owner.latent_key_rope_cache_ptr(),
        position_device_.data(), latent.latent_rank, latent.rope_head_dim,
        stream_.get());
    launch_latent_attention_device({
        .query = {.content = workspace_.latent_query_content_.data(),
                  .rope = workspace_.latent_query_rope_.data()},
        .kv = {.keys = owner.latent_key_cache_ptr(),
               .values = owner.latent_value_cache_ptr(),
               .key_rope = owner.latent_key_rope_cache_ptr()},
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
        .expansion = latent_expansion->data,
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