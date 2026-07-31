#pragma once

// Shared GEMV kernel definitions for the m=1 (decode) path.
//
// Included by:
//   - src/backend/cuda/runtime/gemm_dispatcher.cu  (BF16 decode GEMV)
//   - src/backend/cuda/kernels/linear.cuh           (W8A16 decode GEMV)
//   - src/app/benchmark/cuda/decode_gemv.cu          (isolated microbenchmark)
//
// Defining these once eliminates copy-drift: the benchmark used to carry a
// hand-mirrored copy of each kernel that could silently diverge from the
// production code.

#include <cuda_bf16.h>

// Warp reduce (sum) via shuffle. Static so multiple TUs get independent copies.
static __inline__ __device__ float gemv_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// BF16 weight x BF16 activation GEMV (m=1 decode). One warp per output row,
// 2-wide __nv_bfloat162 loads for coalesced 16-byte accesses.
static __global__ void bf16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ weight,
                                 __nv_bfloat16* __restrict__ y,
                                 int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= n) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(
        weight + static_cast<size_t>(row) * k);
    const int k2 = k >> 1;
    float sum = 0.0f;
    for (int i = lane; i < k2; i += 32) {
        const __nv_bfloat162 xv = x2[i];
        const __nv_bfloat162 wv = w2[i];
        sum += __bfloat162float(xv.x) * __bfloat162float(wv.x) +
               __bfloat162float(xv.y) * __bfloat162float(wv.y);
    }
    sum = gemv_warp_sum(sum);
    if (lane == 0) {
        float value = sum;
        if (k & 1) {
            const int last = k - 1;
            value += __bfloat162float(x[last]) *
                     __bfloat162float(weight[static_cast<size_t>(row) * k + last]);
        }
        if (beta != 0.0f) value += beta * __bfloat162float(y[row]);
        y[row] = __float2bfloat16(value);
    }
}

// INT8 weight x BF16 activation GEMV (W8A16). Supports m=1 (decode) and
// m>1 (batched). One warp per output row. Vectorized: char4 weight loads
// (4 int8 per 4-byte read) + __nv_bfloat162 activation loads for 4 elements
// per lane iteration, keeping both memory paths fully coalesced.
static __global__ void w8a16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                   const int8_t* __restrict__ weight,
                                   const float* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ y,
                                   int m, int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int8_t* row_weight = weight + static_cast<size_t>(output_row) * k;
    const int k4 = k >> 2;
    const char4* w4 = reinterpret_cast<const char4*>(row_weight);

    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;
        const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(input);

        float sum = 0.0f;
        for (int i = lane; i < k4; i += 32) {
            const char4 wv = w4[i];
            const __nv_bfloat162 xv0 = x2[i * 2];
            const __nv_bfloat162 xv1 = x2[i * 2 + 1];
            sum += __bfloat162float(xv0.x) * static_cast<float>(wv.x) +
                   __bfloat162float(xv0.y) * static_cast<float>(wv.y) +
                   __bfloat162float(xv1.x) * static_cast<float>(wv.z) +
                   __bfloat162float(xv1.y) * static_cast<float>(wv.w);
        }
        sum = gemv_warp_sum(sum);

        // Scalar tail for k not divisible by 4.
        const int tail_start = k4 * 4;
        float tail = 0.0f;
        for (int column = tail_start + lane; column < k; column += 32) {
            tail += __bfloat162float(input[column]) *
                    static_cast<float>(row_weight[column]);
        }
        tail = gemv_warp_sum(tail);

        if (lane == 0) {
            float value = (sum + tail) * scales[output_row];
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * __bfloat162float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}
