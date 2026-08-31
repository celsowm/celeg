#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {
namespace {

struct CudaLatentProjectionBuffers {
    const __nv_bfloat16* input = nullptr;
    __nv_bfloat16* query_content = nullptr;
    __nv_bfloat16* query_rope = nullptr;
    __nv_bfloat16* key = nullptr;
    __nv_bfloat16* value = nullptr;
    __nv_bfloat16* key_rope = nullptr;
    int rows = 1;
};

void project_cuda_latent_attention_qkv_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CudaLatentProjectionBuffers& buffers) {
    require_cuda_projected_latent_bindings(attention);
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const int hidden = model.resources_.program_.hidden;
    auto native_fanout = model.native_fanout_scope(
        buffers.input, buffers.rows, hidden);
    model.linear(
        buffers.input, *attention.latent_query, buffers.query_content,
        buffers.rows, layout.latent_query_content_width(), hidden);
    if (layout.latent_query_rope_width() != 0) {
        model.linear(
            buffers.input, *attention.latent_query_rope, buffers.query_rope,
            buffers.rows, layout.latent_query_rope_width(), hidden);
    }
    if (attention.latent_key && attention.latent_value) {
        model.linear(
            buffers.input, *attention.latent_key, buffers.key,
            buffers.rows, latent.latent_rank, hidden);
        model.linear(
            buffers.input, *attention.latent_value, buffers.value,
            buffers.rows, latent.latent_rank, hidden);
        if (attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0) {
            model.linear(
                buffers.input, *attention.latent_key_rope, buffers.key_rope,
                buffers.rows, latent.rope_head_dim, hidden);
        }
    }
}

void project_cuda_prefill_attention_output_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows, int input_width) {
    const int hidden = model.resources_.program_.hidden;
    const bool fuse_residual = model.resources_.options().fused_residuals &&
        !semantics.mixer_norm.after.has_value() &&
        !std::holds_alternative<std::monostate>(semantics.feed_forward);
    model.linear(
        model.workspace_.prefill_op_output_.data(), *attention.out,
        model.workspace_.prefill_hidden_.data(), rows, hidden, input_width,
        fuse_residual ? 1.0f : 0.0f);
    launch_scale(
        model.workspace_.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
}

}

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
    project_cuda_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.normed_.data(),
        .query_content = model.workspace_.latent_query_content_.data(),
        .query_rope = model.workspace_.latent_query_rope_.data(),
        .key = model.workspace_.latent_key_.data(),
        .value = model.workspace_.latent_value_.data(),
        .key_rope = model.workspace_.latent_key_rope_.data(),
        .rows = 1});
}

void project_cuda_prefill_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows) {
    project_cuda_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.prefill_normed_.data(),
        .query_content = model.workspace_.prefill_latent_query_content_.data(),
        .query_rope = model.workspace_.prefill_latent_query_rope_.data(),
        .key = model.workspace_.prefill_latent_key_.data(),
        .value = model.workspace_.prefill_latent_value_.data(),
        .key_rope = model.workspace_.prefill_latent_key_rope_.data(),
        .rows = rows});
}

void project_cuda_prefill_standard_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows) {
    project_cuda_prefill_attention_output_impl(
        model, attention, semantics, rows, attention.layout.query_width());
}

void project_cuda_prefill_latent_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows) {
    project_cuda_prefill_attention_output_impl(
        model, attention, semantics, rows,
        attention.layout.latent_query_content_width());
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
