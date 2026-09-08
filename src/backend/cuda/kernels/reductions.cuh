#pragma once

#include <cfloat>

namespace celeg::cuda_reductions {

static __inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

static __device__ float block_sum(float value, float* warp_sums,
                                  float* block_total) {
    value = warp_sum(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        const int warp_count = (blockDim.x + 31) / 32;
        for (int i = 0; i < warp_count; ++i) total += warp_sums[i];
        *block_total = total;
    }
    __syncthreads();
    return *block_total;
}

static __inline__ __device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

static __device__ float block_max(float value, float* warp_values,
                                  float* block_value) {
    value = warp_max(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -FLT_MAX;
        const int warp_count = (blockDim.x + 31) / 32;
        for (int i = 0; i < warp_count; ++i)
            maximum = fmaxf(maximum, warp_values[i]);
        *block_value = maximum;
    }
    __syncthreads();
    return *block_value;
}

}