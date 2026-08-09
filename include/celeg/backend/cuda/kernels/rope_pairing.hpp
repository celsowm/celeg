#pragma once

#include "celeg/backend/cuda/kernels/rope.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_adjacent_qk_norm_rope_positions(
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const __nv_bfloat16* query_norm,
    const __nv_bfloat16* key_norm,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dim,
    const int32_t* positions,
    float rope_theta,
    float rotary_fraction,
    float epsilon,
    bool normalize,
    CudaRopeScaling scaling,
    cudaStream_t stream);

} // namespace celeg
