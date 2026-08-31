#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"

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

}
