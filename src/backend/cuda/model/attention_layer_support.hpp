#pragma once

#include "detail/layer_state.hpp"
#include "kernels/kernels.cuh"

#include <vector>

namespace celeg {

struct CudaAttentionOwner {
    AttentionLayer* layer = nullptr;
    int model_layer = -1;
};

CudaAttentionOwner resolve_cuda_attention_owner(
    AttentionLayer& attention,
    int layer_index,
    std::vector<Layer>& layers);

Bf16KvView cuda_bf16_kv_view(AttentionLayer& owner);
Int8KvView cuda_int8_kv_view(AttentionLayer& owner);

}
