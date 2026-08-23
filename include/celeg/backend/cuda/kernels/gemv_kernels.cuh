#pragma once


#include <cuda_bf16.h>

// Single source of truth for w8a16_gemv_kernel's block shape: the kernel's
// own indexing math (`warp`, `output_row`) depends on this matching the
// launcher's block dimension exactly. Tried dropping this to 1 (one warp
// per block, since the kernel shares no state -- no smem, no cross-warp
// reduction -- across warps of the same block) to raise SM occupancy on
// narrow (n ~ 1024) decode-time matrices, where `ncu --set full` measured
// only 17% achieved occupancy / 0.13 waves at wpb=8 on a 170-SM GPU. That
// made occupancy *worse* (12.8%) and end-to-end decode throughput did not
// move (134.4 -> 134.9 tok/s, noise): at this kernel's ~5us duration,
// block-dispatch overhead for many tiny 1-warp blocks dominates over the
// SM-fill benefit. Left at 8; see docs/QUANTIZATION_SUPPORT_MATRIX.md GPU
// decode section for the full writeup.
#define W8A16_WARPS_PER_BLOCK 8

static __inline__ __device__ float gemv_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

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

static __global__ void w8a16_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                   const int8_t* __restrict__ weight,
                                   const float* __restrict__ scales,
                                   __nv_bfloat16* __restrict__ y,
                                   int m, int n, int k, float beta) {
    constexpr int warps_per_block = W8A16_WARPS_PER_BLOCK;
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
