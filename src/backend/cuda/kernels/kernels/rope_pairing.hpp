#pragma once

#include "celeg/backend/cuda/kernels/rope.hpp"
#include "celeg/model/definition.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_qk_norm_rope_positions(
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
    RopePairingKind pairing,
    CudaRopeScaling scaling,
    cudaStream_t stream);

inline void launch_adjacent_qk_norm_rope_positions(
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
    cudaStream_t stream) {
    launch_qk_norm_rope_positions(
        query, key, query_norm, key_norm, rows, query_heads, key_value_heads,
        head_dim, positions, rope_theta, rotary_fraction, epsilon, normalize,
        RopePairingKind::AdjacentPairs, scaling, stream);
}

}
