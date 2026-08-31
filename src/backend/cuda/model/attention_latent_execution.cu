#include "detail/compiled_model.hpp"
#include "attention_kv_store.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "backend/cuda/phase_profile.hpp"
#include "kernels/kernels.cuh"
#include "kernels/attention_output.hpp"
#include "kernels/rope_pairing.hpp"
#include "backend/cuda/weight_layout.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_latent_attention(
    Layer& layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<std::size_t>(layer_index));
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention ||
        compiled_attention->execution.kind != AttentionExecutionKind::Latent) {
        throw std::logic_error("CUDA latent attention has an invalid execution descriptor");
    }
    const CompiledAttentionExecution& execution = compiled_attention->execution;
    AttentionLayer* attention = as_attention(layer);
    if (!attention) throw std::logic_error("CUDA layer is not attention");
    if (resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument("CUDA latent attention requires BF16 state storage");
    }
    const AttentionSpec& layout = attention->layout;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        *attention, layer_index, resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    const auto& latent = *layout.latent_state();
    if (layout.output_gate.has_value() || layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA latent attention does not support query gates or M-RoPE yet");
    }
    decode_phase_profile().begin(stream_.get());
    project_cuda_latent_attention_qkv(*this, *attention);
    decode_phase_profile().end(DecodePhase::Projection, stream_.get());
    decode_phase_profile().begin(stream_.get());
    if (const auto* rope = layout.rope_position();
        rope && attention->latent_key_rope && execution.has_decoupled_rope &&
        execution.rotary_width != 0) {
        if (rope->pairing != RopePairingKind::SplitHalf) {
            throw std::invalid_argument(
                "CUDA latent attention requires split-half RoPE pairing");
        }
        launch_dynamic_qk_norm_rope_device(
            workspace_.latent_query_rope_.data(),
            attention->latent_key ? workspace_.latent_key_rope_.data() : nullptr,
            nullptr, nullptr, layout.query_heads, 1,
            latent.rope_head_dim, position_device_.data(),
            static_cast<float>(rope->theta), 1.0f, qk_norm_epsilon, false,
            lower_cuda_rope_scaling(*rope), rope->pairing, stream_.get());
    }
    decode_phase_profile().end(DecodePhase::RopeKv, stream_.get());
    decode_phase_profile().begin(stream_.get());
    store_cuda_latent_kv_contiguous(*this, *attention, owner);
    launch_latent_attention_device({
        .query = {.content = workspace_.latent_query_content_.data(),
                  .rope = layout.latent_query_rope_width() != 0
                              ? workspace_.latent_query_rope_.data() : nullptr},
        .kv = {.keys = owner.latent_key_cache_ptr(),
               .values = owner.latent_value_cache_ptr(),
               .key_rope = owner.latent_key_rope_cache_ptr()},
        .out = workspace_.op_output_.data(),
        .extent = {.position = position_device_.data()},
        .alibi_slopes = attention->alibi_slopes.data(),
        .geometry = {.query_heads = layout.query_heads,
                     .latent_rank = latent.latent_rank,
                     .rotary_width = execution.has_decoupled_rope
                                         ? execution.rotary_width : 0,
                     .score_scale = layout.query_scale,
                     .sliding_window = layout.sliding_window_size()},
        .stream = stream_.get()});
    decode_phase_profile().end(DecodePhase::Attention, stream_.get());
    decode_phase_profile().begin(stream_.get());
    project_latent_attention_output(*attention, semantics);
    decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
}

}
