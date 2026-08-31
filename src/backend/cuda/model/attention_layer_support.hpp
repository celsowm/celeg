#pragma once

#include "detail/layer_state.hpp"
#include "kernels/kernels.cuh"

#include <vector>

namespace celeg {

struct CudaAttentionOwner {
    AttentionLayer* layer = nullptr;
    int model_layer = -1;
};

struct CudaQkvProjectionView {
    __nv_bfloat16* query = nullptr;
    __nv_bfloat16* key = nullptr;
    __nv_bfloat16* value = nullptr;
    int query_projection_width = 0;
};

CudaAttentionOwner resolve_cuda_attention_owner(
    AttentionLayer& attention,
    int layer_index,
    std::vector<Layer>& layers);

CudaQkvProjectionView make_cuda_qkv_projection_view(
    AttentionLayer& attention,
    __nv_bfloat16* storage);

Bf16KvView cuda_bf16_kv_view(AttentionLayer& owner);
Int8KvView cuda_int8_kv_view(AttentionLayer& owner);

}
