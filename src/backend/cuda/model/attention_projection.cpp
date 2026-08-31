#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"
#include "kernels/kernels.cuh"

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

}
