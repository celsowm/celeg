#pragma once

#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/utils.cuh"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>

namespace lfm {
namespace {

__device__ __forceinline__ float bf16_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float rounded_bf16_float(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

__inline__ __device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
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
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
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

__device__ __forceinline__ int unpack_int4(const uint8_t* packed, int column) {
    const uint8_t byte = packed[column >> 1];
    const uint8_t nibble = (column & 1) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16
                        : static_cast<int>(nibble);
}

__host__ __device__ __forceinline__ int attention_threads(int head_dim) {
    int threads = 32;
    while (threads < head_dim && threads < 1024) threads <<= 1;
    return threads;
}

__global__ void head_rmsnorm_kernel(__nv_bfloat16* data,
                                    const __nv_bfloat16* norm_weight,
                                    int rows, int heads, int head_dim,
                                    float eps) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float value = bf16_float(vector[i]);
        sum += value * value;
    }
    __shared__ float warp_sums[32];
    __shared__ float total;
    sum = block_sum(sum, warp_sums, &total);
    __shared__ float inv;
    if (threadIdx.x == 0) inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(vector[i]) * inv);
        vector[i] = __float2bfloat16(normalized * bf16_float(norm_weight[i]));
    }
}

} // namespace
} // namespace lfm
