#include "detail/compiled_model.hpp"
#include "celeg/backend/cuda/attention_norm.hpp"
#include "celeg/backend/cuda/phase_profile.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/kernels/attention_output.hpp"
#include "celeg/backend/cuda/kernels/rope_pairing.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"

#include <stdexcept>

namespace celeg {

PhaseProfile& decode_phase_profile();

void CudaCompiledModel::enqueue_decode_latent_attention(
    Layer& layer, LayerCommon&, int layer_index) {
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
    if (resources_.options_.kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument("CUDA latent attention requires BF16 state storage");
    }
    const AttentionSpec& layout = attention->layout;
    const float qk_norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon : resources_.program_.final_norm.epsilon;
    const bool fuse_mixer_residual = resources_.options_.fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    AttentionLayer* owner = attention;
    if (attention->kv_owner_layer >= 0) {
        owner = as_attention(resources_.layers_.at(
            static_cast<std::size_t>(attention->kv_owner_layer)));
        if (!owner) throw std::logic_error("CUDA shared KV owner is not attention");
    }
    const auto& latent = *layout.latent_state();
    if (layout.output_gate.has_value() || layout.multi_axis_position()) {
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
        if (attention->latent_key_rope && execution.has_decoupled_rope &&
            execution.rotary_width != 0) {
            linear(workspace_.normed_.data(), *attention->latent_key_rope,
                   workspace_.latent_key_rope_.data(), 1,
                   latent.rope_head_dim, resources_.program_.hidden);
        }
    }
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
    if (attention->latent_key && attention->latent_value) {
        launch_store_latent_device(
            workspace_.latent_key_.data(), workspace_.latent_value_.data(),
            attention->latent_key_rope && execution.has_decoupled_rope &&
            execution.rotary_width != 0
                ? workspace_.latent_key_rope_.data() : nullptr,
            owner->latent_key_cache_ptr(), owner->latent_value_cache_ptr(),
            owner->latent_key_rope_cache_ptr(), position_device_.data(),
            latent.latent_rank,
            execution.has_decoupled_rope ? execution.rotary_width : 0,
            stream_.get());
    }
    launch_latent_attention_device({
        .query = {.content = workspace_.latent_query_content_.data(),
                  .rope = layout.latent_query_rope_width() != 0
                              ? workspace_.latent_query_rope_.data() : nullptr},
        .kv = {.keys = owner->latent_key_cache_ptr(),
               .values = owner->latent_value_cache_ptr(),
               .key_rope = owner->latent_key_rope_cache_ptr()},
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
    linear(workspace_.op_output_.data(), *attention->out,
           workspace_.hidden_.data(), 1, resources_.program_.hidden,
           layout.latent_query_content_width(), fuse_mixer_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
    decode_phase_profile().end(DecodePhase::AttnOut, stream_.get());
}

}
