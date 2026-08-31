#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {

CudaQkvProjectionView CudaCompiledModel::project_standard_attention_qkv(
    AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    CudaQkvProjectionView qkv = make_cuda_qkv_projection_view(
        attention, workspace_.qkv_output_.data());
    auto native_fanout = native_fanout_scope(
        workspace_.normed_.data(), 1, resources_.program_.hidden);
    linear(workspace_.normed_.data(), *attention.query, qkv.query,
           1, qkv.query_projection_width, resources_.program_.hidden);
    if (attention.key && attention.value) {
        linear(workspace_.normed_.data(), *attention.key, qkv.key,
               1, layout.key_value_width(), resources_.program_.hidden);
        linear(workspace_.normed_.data(), *attention.value, qkv.value,
               1, layout.key_value_width(), resources_.program_.hidden);
    }
    return qkv;
}

void require_cuda_projected_latent_bindings(const AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    if (!attention.latent_query || !attention.out) {
        throw std::logic_error(
            "CUDA projected latent attention has incomplete query/output bindings");
    }
    if (layout.latent_query_rope_width() != 0 && !attention.latent_query_rope) {
        throw std::logic_error(
            "CUDA projected latent attention is missing the query RoPE projection");
    }
    if ((attention.latent_key == nullptr) != (attention.latent_value == nullptr)) {
        throw std::logic_error(
            "CUDA projected latent attention must bind latent key/value together");
    }
}

void project_cuda_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention) {
    require_cuda_projected_latent_bindings(attention);
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    auto native_fanout = model.native_fanout_scope(
        model.workspace_.normed_.data(), 1, model.resources_.program_.hidden);
    model.linear(
        model.workspace_.normed_.data(), *attention.latent_query,
        model.workspace_.latent_query_content_.data(), 1,
        layout.latent_query_content_width(), model.resources_.program_.hidden);
    if (layout.latent_query_rope_width() != 0) {
        model.linear(
            model.workspace_.normed_.data(), *attention.latent_query_rope,
            model.workspace_.latent_query_rope_.data(), 1,
            layout.latent_query_rope_width(), model.resources_.program_.hidden);
    }
    if (attention.latent_key && attention.latent_value) {
        model.linear(
            model.workspace_.normed_.data(), *attention.latent_key,
            model.workspace_.latent_key_.data(), 1, latent.latent_rank,
            model.resources_.program_.hidden);
        model.linear(
            model.workspace_.normed_.data(), *attention.latent_value,
            model.workspace_.latent_value_.data(), 1, latent.latent_rank,
            model.resources_.program_.hidden);
        if (attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0) {
            model.linear(
                model.workspace_.normed_.data(), *attention.latent_key_rope,
                model.workspace_.latent_key_rope_.data(), 1,
                latent.rope_head_dim, model.resources_.program_.hidden);
        }
    }
}

void CudaCompiledModel::project_standard_attention_output(
    AttentionLayer& attention, const CompiledLayerProgram& semantics) {
    const bool fuse_residual = resources_.options().fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, attention.layout.query_width(),
           fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

void CudaCompiledModel::project_latent_attention_output(
    AttentionLayer& attention, const CompiledLayerProgram& semantics) {
    const bool fuse_residual = resources_.options().fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden,
           attention.layout.latent_query_content_width(),
           fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

}
