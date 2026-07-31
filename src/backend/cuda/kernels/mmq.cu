#include "celeg/backend/cuda/kernels/mmq.hpp"
#include "kernel_common.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <cstring>

namespace celeg {
namespace {

// Mirrors gguf.cu's BlockQ4K layout (super-block layout is GGUF's on-disk
// format, not something this file gets to choose).
struct BlockQ4K {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

using celeg::gguf_blocks::q4k_scale_min;

constexpr int kSuperBlock = 256;
constexpr int kSubBlocksPerSuperBlock = kSuperBlock / kMmqQ8_1BlockSize;

__global__ void quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x,
                                     int8_t* __restrict__ q8,
                                     float* __restrict__ scales,
                                     float* __restrict__ sums, int k) {
    const int row = blockIdx.y;
    const int block_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const size_t base = static_cast<size_t>(row) * k +
                        static_cast<size_t>(block_idx) * kMmqQ8_1BlockSize;
    const float v = bf16_float(x[base + lane]);

    float absmax = fabsf(v);
    for (int offset = 16; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
    }
    absmax = __shfl_sync(0xffffffffu, absmax, 0);
    const float delta = absmax > 0.0f ? absmax / 127.0f : 1.0f;

    int qi = __float2int_rn(v / delta);
    qi = max(-127, min(127, qi));
    q8[base + lane] = static_cast<int8_t>(qi);

    int sum = qi;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const size_t block_count = static_cast<size_t>(k) / kMmqQ8_1BlockSize;
        const size_t block_index =
            static_cast<size_t>(row) * block_count + block_idx;
        scales[block_index] = delta;
        sums[block_index] = static_cast<float>(sum);
    }
}

__global__ void q4k_mmq_kernel(const int8_t* __restrict__ q8,
                               const float* __restrict__ q8_scales,
                               const float* __restrict__ q8_sums,
                               const uint8_t* __restrict__ blocks,
                               __nv_bfloat16* __restrict__ y, int m, int n,
                               int k, size_t row_bytes, int output_stride,
                               float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int sub_blocks_total = k / kMmqQ8_1BlockSize;
    const uint8_t* row_blocks = blocks + static_cast<size_t>(output_row) * row_bytes;

    for (int activation_row = blockIdx.y; activation_row < m;
         activation_row += gridDim.y) {
        const int8_t* a_q8 = q8 + static_cast<size_t>(activation_row) * k;
        const float* a_scale =
            q8_scales + static_cast<size_t>(activation_row) * sub_blocks_total;
        const float* a_sum =
            q8_sums + static_cast<size_t>(activation_row) * sub_blocks_total;

        float acc = 0.0f;
        for (int sub = lane; sub < sub_blocks_total; sub += 32) {
            const int superblock = sub / kSubBlocksPerSuperBlock;
            const int within = sub % kSubBlocksPerSuperBlock;
            const BlockQ4K* blk =
                reinterpret_cast<const BlockQ4K*>(row_blocks) + superblock;
            uint8_t sc = 0, mn = 0;
            q4k_scale_min(within, blk->scales, sc, mn);
            const float d_w = __half2float(blk->d);
            const float dmin_w = __half2float(blk->dmin);
            const uint8_t* qs = blk->qs + (within >> 1) * 32;
            const bool high = (within & 1) != 0;
            const int8_t* a_ptr = a_q8 + sub * kMmqQ8_1BlockSize;

            int dot = 0;
#pragma unroll
            for (int g = 0; g < kMmqQ8_1BlockSize; g += 4) {
                int packed_w = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint8_t byte = qs[g + t];
                    const int wv = high ? (byte >> 4) : (byte & 0xF);
                    packed_w |= wv << (t * 8);
                }
                int packed_a = 0;
                memcpy(&packed_a, a_ptr + g, 4);
                dot = __dp4a(packed_w, packed_a, dot);
            }

            const float delta_a = a_scale[sub];
            const float sum_a = a_sum[sub];
            acc += delta_a * (d_w * sc * static_cast<float>(dot) - dmin_w * mn * sum_a);
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            float value = acc;
            const size_t out_index =
                static_cast<size_t>(activation_row) * output_stride + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[out_index]);
            y[out_index] = __float2bfloat16(value);
        }
    }
}

// ---------------------------------------------------------------------------
// Q6_K MMQ: same __dp4a strategy as Q4_K, adapted for the Q6_K super-block.
//
// Q6_K super-block (256 elements, 210 bytes):
//   uint8  ql[128];      // lower 4 bits of each 6-bit value
//   uint8  qh[64];       // upper 2 bits, 4 elements packed per byte
//   int8   scales[16];   // per 16-element sub-block scale
//   half   d;            // super-block scale
// Dequant: y = d * scales[is] * (q - 32), q is 6-bit unsigned (0..63).
//
// Each 32-element Q8_1 activation block spans two Q6_K scale sub-blocks
// (16 elements each), so the dot product must be split into two halves
// with separate scale application. The -32 weight offset is folded in via
// the per-16-element activation sum, computed alongside the dp4a using
// __dp4a(0x01010101, packed_a, sum).
// ---------------------------------------------------------------------------

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};

constexpr int kQ6KSuperBlock = 256;
constexpr int kQ6KSubBlocksPerSuperBlock = kQ6KSuperBlock / kMmqQ8_1BlockSize;

__global__ void q6k_mmq_kernel(const int8_t* __restrict__ q8,
                               const float* __restrict__ q8_scales,
                               const float* __restrict__ q8_sums,
                               const uint8_t* __restrict__ blocks,
                               __nv_bfloat16* __restrict__ y, int m, int n,
                               int k, size_t row_bytes, int output_stride,
                               float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int sub_blocks_total = k / kMmqQ8_1BlockSize;
    const uint8_t* row_blocks = blocks + static_cast<size_t>(output_row) * row_bytes;

    for (int activation_row = blockIdx.y; activation_row < m;
         activation_row += gridDim.y) {
        const int8_t* a_q8 = q8 + static_cast<size_t>(activation_row) * k;
        const float* a_scale =
            q8_scales + static_cast<size_t>(activation_row) * sub_blocks_total;

        float acc = 0.0f;
        for (int sub = lane; sub < sub_blocks_total; sub += 32) {
            const int superblock = sub / kQ6KSubBlocksPerSuperBlock;
            const int within = sub % kQ6KSubBlocksPerSuperBlock;
            const BlockQ6K* blk =
                reinterpret_cast<const BlockQ6K*>(row_blocks) + superblock;

            const int half = within >> 2;        // 0 or 1
            const int grp = within & 3;           // 0..3
            const int is_lo = half * 8 + grp * 2;
            const int is_hi = is_lo + 1;
            const float d = __half2float(blk->d);
            const float sc_lo = static_cast<float>(blk->scales[is_lo]);
            const float sc_hi = static_cast<float>(blk->scales[is_hi]);

            const uint8_t* ql = blk->ql + half * 64;
            const uint8_t* qh = blk->qh + half * 32;
            const int8_t* a_ptr = a_q8 + sub * kMmqQ8_1BlockSize;
            const float delta_a = a_scale[sub];

            int dot_lo = 0, dot_hi = 0;
            int sum_lo = 0, sum_hi = 0;

#pragma unroll
            for (int g = 0; g < 16; g += 4) {
                int packed_w = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int n = g + t;
                    int q;
                    if (grp == 0) {
                        q = (ql[n] & 0xF) | (((qh[n] >> 0) & 3) << 4);
                    } else if (grp == 1) {
                        q = (ql[n + 32] & 0xF) | (((qh[n] >> 2) & 3) << 4);
                    } else if (grp == 2) {
                        q = (ql[n] >> 4) | (((qh[n] >> 4) & 3) << 4);
                    } else {
                        q = (ql[n + 32] >> 4) | (((qh[n] >> 6) & 3) << 4);
                    }
                    packed_w |= q << (t * 8);
                }
                int packed_a;
                memcpy(&packed_a, a_ptr + g, 4);
                dot_lo = __dp4a(packed_w, packed_a, dot_lo);
                sum_lo = __dp4a(0x01010101, packed_a, sum_lo);
            }

#pragma unroll
            for (int g = 16; g < 32; g += 4) {
                int packed_w = 0;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int n = g + t;
                    int q;
                    if (grp == 0) {
                        q = (ql[n] & 0xF) | (((qh[n] >> 0) & 3) << 4);
                    } else if (grp == 1) {
                        q = (ql[n + 32] & 0xF) | (((qh[n] >> 2) & 3) << 4);
                    } else if (grp == 2) {
                        q = (ql[n] >> 4) | (((qh[n] >> 4) & 3) << 4);
                    } else {
                        q = (ql[n + 32] >> 4) | (((qh[n] >> 6) & 3) << 4);
                    }
                    packed_w |= q << (t * 8);
                }
                int packed_a;
                memcpy(&packed_a, a_ptr + g, 4);
                dot_hi = __dp4a(packed_w, packed_a, dot_hi);
                sum_hi = __dp4a(0x01010101, packed_a, sum_hi);
            }

            acc += d * delta_a * (sc_lo * static_cast<float>(dot_lo - 32 * sum_lo) +
                                  sc_hi * static_cast<float>(dot_hi - 32 * sum_hi));
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            float value = acc;
            const size_t out_index =
                static_cast<size_t>(activation_row) * output_stride + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[out_index]);
            y[out_index] = __float2bfloat16(value);
        }
    }
}

} // namespace

void launch_quantize_q8_1(const __nv_bfloat16* x, int8_t* q8, float* scales,
                          float* sums, int rows, int k, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(k / kMmqQ8_1BlockSize),
                    static_cast<unsigned>(rows));
    quantize_q8_1_kernel<<<grid, kMmqQ8_1BlockSize, 0, stream>>>(
        x, q8, scales, sums, k);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_q4k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    q4k_mmq_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_q6k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    q6k_mmq_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
