#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace celeg {

void launch_orthogonalize_current_value(
    __nv_bfloat16* output,
    const __nv_bfloat16* current_value,
    int rows,
    int query_heads,
    int key_value_heads,
    int head_dim,
    float minimum_norm_squared,
    cudaStream_t stream);

} // namespace celeg
