#pragma once

#include <array>
#include <cstdint>
#include <cuda_bf16.h>

namespace celeg {

struct AttentionLayer;
struct CudaCompiledModel;

void prepare_cuda_token_attention_qk(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    bool paged,
    const std::array<int32_t, 3>* rope_position);

}
