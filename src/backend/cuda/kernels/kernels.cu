#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/utils.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace lfm {
namespace {

// This file is the single CUDA translation-unit boundary.  Each included
// implementation unit owns one kernel concern so launchers remain linked
// together without duplicating the shared device helpers above.

__device__ __forceinline__ float bf16_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float rounded_bf16_float(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float* warp_sums, float* block_total) {
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

__inline__ __device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ float block_max(float value, float* warp_values, float* block_value) {
    value = warp_max(value);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = 0.0f;
        const int warp_count = (blockDim.x + 31) / 32;
        for (int i = 0; i < warp_count; ++i) maximum = fmaxf(maximum, warp_values[i]);
        *block_value = maximum;
    }
    __syncthreads();
    return *block_value;
}

__device__ __forceinline__ int8_t quantize_symmetric_int8(float value, float scale) {
    const float normalized = scale > 0.0f ? value / scale : 0.0f;
    const int rounded = static_cast<int>(lrintf(normalized));
    return static_cast<int8_t>(max(-127, min(127, rounded)));
}

__device__ __forceinline__ int resolved_position(int value,
                                                  const int32_t* pointer,
                                                  bool use_pointer) {
    return use_pointer ? *pointer : value;
}

#include "embedding.inl"
#include "transform.inl"
#include "attention.inl"
#include "sampling.inl"
#include "packed.inl"

} // namespace lfm
