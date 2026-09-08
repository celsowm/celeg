#pragma once

#include "kernels/kernels.cuh"
#include "reductions.cuh"
#include "utils.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace celeg {
namespace {

using cuda_reductions::block_max;
using cuda_reductions::block_sum;
using cuda_reductions::warp_max;
using cuda_reductions::warp_sum;

__device__ __forceinline__ float bf16_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float rounded_bf16_float(float value) {
    return __bfloat162float(__float2bfloat16(value));
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

template <typename... Ptrs>
__device__ __forceinline__ bool bf16x2_aligned(Ptrs... ptrs) {
    return (((reinterpret_cast<uintptr_t>(ptrs) & 3u) == 0) && ...);
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
    if ((head_dim & 1) == 0 && bf16x2_aligned(vector, norm_weight)) {
        const int half = head_dim >> 1;
        __nv_bfloat162* vector2 = reinterpret_cast<__nv_bfloat162*>(vector);
        const __nv_bfloat162* weight2 = reinterpret_cast<const __nv_bfloat162*>(norm_weight);
        for (int i = threadIdx.x; i < half; i += blockDim.x) {
            const __nv_bfloat162 vv = vector2[i];
            const __nv_bfloat162 wv = weight2[i];
            const float n0 = rounded_bf16_float(__low2float(vv) * inv);
            const float n1 = rounded_bf16_float(__high2float(vv) * inv);
            vector2[i] = __floats2bfloat162_rn(n0 * __low2float(wv), n1 * __high2float(wv));
        }
    } else {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float normalized = rounded_bf16_float(bf16_float(vector[i]) * inv);
            vector[i] = __float2bfloat16(normalized * bf16_float(norm_weight[i]));
        }
    }
}

}
}
