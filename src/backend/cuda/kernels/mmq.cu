#include "celeg/backend/cuda/kernels/mmq.hpp"
#include "kernel_common.cuh"
#include "celeg/checkpoint/gguf_blocks.hpp"

#include <cstring>
#include <cstdlib>
#include <mma.h>
#include <mutex>
#include <unordered_map>

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
// A prefill tile keeps one GGUF weight row in registers while accumulating
// several activation rows. The previous kernel launched one block for every
// (activation,row-of-W) pair, which was acceptable for decode but made native
// prefill overwhelmingly launch- and weight-read-bound.
constexpr int kMmqPrefillTileRows = 8;
constexpr int kMmqTensorCoreTile = 16;

bool mmq_tensor_core_supported() {
    // int8 mma.sync (nvcuda::wmma with signed-char fragments) requires
    // sm_72+. The result is cached by device, not process-wide: callers may
    // legally switch the current CUDA device between model operations.
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static std::mutex mutex;
    static std::unordered_map<int, bool> supported_by_device;
    std::lock_guard lock(mutex);
    if (const auto it = supported_by_device.find(device);
        it != supported_by_device.end()) {
        return it->second;
    }
    int major = 0;
    int minor = 0;
    const bool supported =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
        (major * 10 + minor) >= 72;
    supported_by_device.emplace(device, supported);
    return supported;
}

bool mmq_tensor_core_enabled() {
    // The randomized Q4_K/Q6_K reference suite (see cuda_gguf_kernels_test.cu)
    // cross-checks the MMA prefill kernels exactly against the validated
    // decode kernel, so tensor cores are the default whenever the GPU
    // supports int8 mma.sync. CELEG_MMQ_TENSOR_CORES is left as an explicit
    // override only so the same binary can still bisect a regression between
    // the MMA and DP4A prefill kernels; it is not meant to be user-facing.
    //
    // This resolves once per (thread, device): every matmul launch used to
    // pay a getenv() plus mmq_tensor_core_supported()'s mutex+map lookup, on
    // the decode hot path that's ~one call per GEMM per layer per token. The
    // thread_local pair below skips straight to the cached answer whenever
    // the device hasn't changed since this thread last resolved it; a device
    // switch (legal per mmq_tensor_core_supported()'s contract) just falls
    // through to a full, still-correct re-resolution.
    thread_local int cached_device = -1;
    thread_local bool cached_enabled = false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (device == cached_device) return cached_enabled;

    const char* setting = std::getenv("CELEG_MMQ_TENSOR_CORES");
    cached_enabled = setting != nullptr ? setting[0] == '1' : mmq_tensor_core_supported();
    cached_device = device;
    return cached_enabled;
}

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
                    const int index = g + t;
                    int q;
                    if (grp == 0) {
                        q = (ql[index] & 0xF) | (((qh[index] >> 0) & 3) << 4);
                    } else if (grp == 1) {
                        q = (ql[index + 32] & 0xF) | (((qh[index] >> 2) & 3) << 4);
                    } else if (grp == 2) {
                        q = (ql[index] >> 4) | (((qh[index] >> 4) & 3) << 4);
                    } else {
                        q = (ql[index + 32] >> 4) | (((qh[index] >> 6) & 3) << 4);
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
                    const int index = g + t;
                    int q;
                    if (grp == 0) {
                        q = (ql[index] & 0xF) | (((qh[index] >> 0) & 3) << 4);
                    } else if (grp == 1) {
                        q = (ql[index + 32] & 0xF) | (((qh[index] >> 2) & 3) << 4);
                    } else if (grp == 2) {
                        q = (ql[index] >> 4) | (((qh[index] >> 4) & 3) << 4);
                    } else {
                        q = (ql[index + 32] >> 4) | (((qh[index] >> 6) & 3) << 4);
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

template <int TileRows>
__global__ void q4k_mmq_prefill_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ q8_scales,
    const float* __restrict__ q8_sums, const uint8_t* __restrict__ blocks,
    __nv_bfloat16* __restrict__ y, int m, int n, int k, size_t row_bytes,
    int output_stride, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int activation_base = blockIdx.y * TileRows;
    if (output_row >= n || activation_base >= m) return;

    const int sub_blocks_total = k / kMmqQ8_1BlockSize;
    const uint8_t* row_blocks = blocks + static_cast<size_t>(output_row) * row_bytes;
    float acc[TileRows] = {};
    for (int sub = lane; sub < sub_blocks_total; sub += 32) {
        const int superblock = sub / kSubBlocksPerSuperBlock;
        const int within = sub % kSubBlocksPerSuperBlock;
        const BlockQ4K* blk = reinterpret_cast<const BlockQ4K*>(row_blocks) + superblock;
        uint8_t sc = 0, mn = 0;
        q4k_scale_min(within, blk->scales, sc, mn);
        const uint8_t* qs = blk->qs + (within >> 1) * 32;
        const bool high = (within & 1) != 0;
        int dots[TileRows] = {};
#pragma unroll
        for (int g = 0; g < kMmqQ8_1BlockSize; g += 4) {
            int packed_w = 0;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint8_t byte = qs[g + t];
                packed_w |= static_cast<int>(high ? (byte >> 4) : (byte & 0xF)) << (t * 8);
            }
#pragma unroll
            for (int r = 0; r < TileRows; ++r) {
                if (activation_base + r < m) {
                    int packed_a;
                    const int8_t* a = q8 + static_cast<size_t>(activation_base + r) * k +
                                      sub * kMmqQ8_1BlockSize + g;
                    memcpy(&packed_a, a, sizeof(packed_a));
                    dots[r] = __dp4a(packed_w, packed_a, dots[r]);
                }
            }
        }
        const float weight_scale = __half2float(blk->d) * static_cast<float>(sc);
        const float weight_min = __half2float(blk->dmin) * static_cast<float>(mn);
#pragma unroll
        for (int r = 0; r < TileRows; ++r) {
            if (activation_base + r < m) {
                const size_t activation_offset =
                    static_cast<size_t>(activation_base + r) * sub_blocks_total + sub;
                // q8_sums holds the sum of the *quantized* int8 activations, so
                // the min correction is on the same scale as the dot product:
                // the activation delta multiplies both terms.
                acc[r] += q8_scales[activation_offset] *
                          (weight_scale * static_cast<float>(dots[r]) -
                           weight_min * q8_sums[activation_offset]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < TileRows; ++r) {
        if (activation_base + r < m) {
            const float reduced = warp_sum(acc[r]);
            if (lane == 0) {
                const size_t index = static_cast<size_t>(activation_base + r) * output_stride + output_row;
                float value = reduced;
                if (beta != 0.0f) value += beta * bf16_float(y[index]);
                y[index] = __float2bfloat16(value);
            }
        }
    }
}

// Ampere path for real prefill. Each warp owns a 16x16 output tile and uses
// integer tensor cores for each 32-wide GGUF sub-block; WarpsPerBlock warps
// share one block, all covering the same 16 activation rows (blockIdx.y), so
// the 16x32 activation tile is loaded once per block and reused by every
// warp instead of being reloaded per output tile. This keeps enough resident
// warps per SM to hide wmma/global-load latency instead of running one warp
// (32 threads) per block. Q4_K's scale/min are per 32 values, so the two
// k=16 MMA results are combined before applying the floating-point
// dequantization correction. The raw GGUF blocks remain the only persistent
// weight storage.
template <int WarpsPerBlock>
__global__ void q4k_mmq_tensor_core_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ q8_scales,
    const float* __restrict__ q8_sums, const uint8_t* __restrict__ blocks,
    __nv_bfloat16* __restrict__ y, int m, int n, int k, size_t row_bytes,
    int output_stride, float beta) {
    namespace wmma = nvcuda::wmma;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int activation_base = blockIdx.y * kMmqTensorCoreTile;
    const int output_base = (blockIdx.x * WarpsPerBlock + warp) * kMmqTensorCoreTile;
    __shared__ int8_t activation_tile[kMmqTensorCoreTile * kMmqQ8_1BlockSize];
    __shared__ int8_t weight_tile[WarpsPerBlock][kMmqQ8_1BlockSize * kMmqTensorCoreTile];
    __shared__ int dot_lo_tile[WarpsPerBlock][kMmqTensorCoreTile * kMmqTensorCoreTile];
    __shared__ int dot_hi_tile[WarpsPerBlock][kMmqTensorCoreTile * kMmqTensorCoreTile];
    float accumulated[8] = {};

    for (int sub = 0; sub < k / kMmqQ8_1BlockSize; ++sub) {
        for (int index = threadIdx.x; index < kMmqTensorCoreTile * kMmqQ8_1BlockSize; index += blockDim.x) {
            const int row = index / kMmqQ8_1BlockSize;
            const int column = index % kMmqQ8_1BlockSize;
            activation_tile[index] = activation_base + row < m
                ? q8[static_cast<size_t>(activation_base + row) * k +
                     sub * kMmqQ8_1BlockSize + column]
                : 0;
        }
        for (int index = lane; index < kMmqTensorCoreTile * kMmqQ8_1BlockSize; index += 32) {
            const int output = index / kMmqQ8_1BlockSize;
            const int column = index % kMmqQ8_1BlockSize;
            int value = 0;
            if (output_base + output < n) {
                const BlockQ4K* block = reinterpret_cast<const BlockQ4K*>(
                    blocks + static_cast<size_t>(output_base + output) * row_bytes) + sub / 8;
                const int within = sub & 7;
                const uint8_t byte = block->qs[(within >> 1) * 32 + column];
                value = (within & 1) ? (byte >> 4) : (byte & 0xF);
            }
            // B is [K,N] in column-major form for C=A*B. Its leading
            // dimension is the full 32-wide GGUF sub-block, not N.
            weight_tile[warp][output * kMmqQ8_1BlockSize + column] = static_cast<int8_t>(value);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
        wmma::fill_fragment(c, 0);
        wmma::load_matrix_sync(a, activation_tile, kMmqQ8_1BlockSize);
        wmma::load_matrix_sync(b, weight_tile[warp], kMmqQ8_1BlockSize);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(dot_lo_tile[warp], c, kMmqTensorCoreTile, wmma::mem_row_major);
        wmma::fill_fragment(c, 0);
        wmma::load_matrix_sync(a, activation_tile + 16, kMmqQ8_1BlockSize);
        wmma::load_matrix_sync(b, weight_tile[warp] + 16, kMmqQ8_1BlockSize);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(dot_hi_tile[warp], c, kMmqTensorCoreTile, wmma::mem_row_major);
        __syncthreads();
        for (int index = lane; index < kMmqTensorCoreTile * kMmqTensorCoreTile; index += 32) {
            const int row = index / kMmqTensorCoreTile;
            const int output = index % kMmqTensorCoreTile;
            if (activation_base + row < m && output_base + output < n) {
                const BlockQ4K* block = reinterpret_cast<const BlockQ4K*>(
                    blocks + static_cast<size_t>(output_base + output) * row_bytes) + sub / 8;
                uint8_t scale = 0, minimum = 0;
                q4k_scale_min(sub & 7, block->scales, scale, minimum);
                const size_t activation_offset =
                    static_cast<size_t>(activation_base + row) * (k / kMmqQ8_1BlockSize) + sub;
                // store_matrix_sync used mem_row_major with ldm=16, so the
                // accumulator tile is indexed [row][output] — the same order
                // this loop derives from `index`. The activation delta scales
                // both the dot product and the min correction, because
                // q8_sums holds the sum of the quantized int8 activations.
                accumulated[index / 32] += q8_scales[activation_offset] *
                    (__half2float(block->d) * static_cast<float>(scale) *
                         static_cast<float>(dot_lo_tile[warp][index] + dot_hi_tile[warp][index]) -
                     __half2float(block->dmin) * static_cast<float>(minimum) *
                         q8_sums[activation_offset]);
            }
        }
        __syncthreads();
    }
    for (int index = lane; index < kMmqTensorCoreTile * kMmqTensorCoreTile; index += 32) {
        const int row = index / kMmqTensorCoreTile;
        const int output = index % kMmqTensorCoreTile;
        if (activation_base + row < m && output_base + output < n) {
            const size_t destination = static_cast<size_t>(activation_base + row) * output_stride + output_base + output;
            float value = accumulated[index / 32];
            if (beta != 0.0f) value += beta * bf16_float(y[destination]);
            y[destination] = __float2bfloat16(value);
        }
    }
}

__device__ __forceinline__ int q6k_pack4(const BlockQ6K* blk, int half, int group,
                                          int first) {
    const uint8_t* ql = blk->ql + half * 64;
    const uint8_t* qh = blk->qh + half * 32;
    int packed = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int index = first + t;
        int value = 0;
        if (group == 0) value = (ql[index] & 0xF) | (((qh[index] >> 0) & 3) << 4);
        else if (group == 1) value = (ql[index + 32] & 0xF) | (((qh[index] >> 2) & 3) << 4);
        else if (group == 2) value = (ql[index] >> 4) | (((qh[index] >> 4) & 3) << 4);
        else value = (ql[index + 32] >> 4) | (((qh[index] >> 6) & 3) << 4);
        packed |= value << (t * 8);
    }
    return packed;
}

template <int WarpsPerBlock>
__global__ void q6k_mmq_tensor_core_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ q8_scales,
    const float* __restrict__ q8_sums, const uint8_t* __restrict__ blocks,
    __nv_bfloat16* __restrict__ y, int m, int n, int k, size_t row_bytes,
    int output_stride, float beta) {
    namespace wmma = nvcuda::wmma;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int activation_base = blockIdx.y * kMmqTensorCoreTile;
    const int output_base = (blockIdx.x * WarpsPerBlock + warp) * kMmqTensorCoreTile;
    __shared__ int8_t activation_tile[kMmqTensorCoreTile * kMmqQ8_1BlockSize];
    __shared__ int8_t weight_tile[WarpsPerBlock][kMmqQ8_1BlockSize * kMmqTensorCoreTile];
    __shared__ int dot_lo_tile[WarpsPerBlock][kMmqTensorCoreTile * kMmqTensorCoreTile];
    __shared__ int dot_hi_tile[WarpsPerBlock][kMmqTensorCoreTile * kMmqTensorCoreTile];
    float accumulated[8] = {};

    for (int sub = 0; sub < k / kMmqQ8_1BlockSize; ++sub) {
        for (int index = threadIdx.x; index < kMmqTensorCoreTile * kMmqQ8_1BlockSize; index += blockDim.x) {
            const int row = index / kMmqQ8_1BlockSize;
            const int column = index % kMmqQ8_1BlockSize;
            activation_tile[index] = activation_base + row < m
                ? q8[static_cast<size_t>(activation_base + row) * k + sub * kMmqQ8_1BlockSize + column]
                : 0;
        }
        for (int index = lane; index < kMmqTensorCoreTile * kMmqQ8_1BlockSize; index += 32) {
            const int output = index / kMmqQ8_1BlockSize;
            const int column = index % kMmqQ8_1BlockSize;
            int value = 0;
            if (output_base + output < n) {
                const BlockQ6K* block = reinterpret_cast<const BlockQ6K*>(
                    blocks + static_cast<size_t>(output_base + output) * row_bytes) + sub / 8;
                const int within = sub & 7;
                const int half = within >> 2;
                const int group = within & 3;
                const int index_in_group = column;
                const uint8_t* ql = block->ql + half * 64;
                const uint8_t* qh = block->qh + half * 32;
                if (group == 0) value = (ql[index_in_group] & 0xF) | (((qh[index_in_group] >> 0) & 3) << 4);
                else if (group == 1) value = (ql[index_in_group + 32] & 0xF) | (((qh[index_in_group] >> 2) & 3) << 4);
                else if (group == 2) value = (ql[index_in_group] >> 4) | (((qh[index_in_group] >> 4) & 3) << 4);
                else value = (ql[index_in_group + 32] >> 4) | (((qh[index_in_group] >> 6) & 3) << 4);
            }
            weight_tile[warp][output * kMmqQ8_1BlockSize + column] = static_cast<int8_t>(value);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> c;
        wmma::fill_fragment(c, 0);
        wmma::load_matrix_sync(a, activation_tile, kMmqQ8_1BlockSize);
        wmma::load_matrix_sync(b, weight_tile[warp], kMmqQ8_1BlockSize);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(dot_lo_tile[warp], c, kMmqTensorCoreTile, wmma::mem_row_major);
        wmma::fill_fragment(c, 0);
        wmma::load_matrix_sync(a, activation_tile + 16, kMmqQ8_1BlockSize);
        wmma::load_matrix_sync(b, weight_tile[warp] + 16, kMmqQ8_1BlockSize);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(dot_hi_tile[warp], c, kMmqTensorCoreTile, wmma::mem_row_major);
        __syncthreads();
        for (int index = lane; index < kMmqTensorCoreTile * kMmqTensorCoreTile; index += 32) {
            const int row = index / kMmqTensorCoreTile;
            const int output = index % kMmqTensorCoreTile;
            if (activation_base + row < m && output_base + output < n) {
                const BlockQ6K* block = reinterpret_cast<const BlockQ6K*>(
                    blocks + static_cast<size_t>(output_base + output) * row_bytes) + sub / 8;
                const int within = sub & 7;
                const int half = within >> 2;
                const int group = within & 3;
                const int scale_lo_index = half * 8 + group * 2;
                const size_t activation_offset =
                    static_cast<size_t>(activation_base + row) * (k / kMmqQ8_1BlockSize) + sub;
                // Q6_K changes scale every 16 elements, so derive the two raw
                // activation sums locally from the shared Q8 tile.
                int sum_lo = 0, sum_hi = 0;
#pragma unroll
                for (int cidx = 0; cidx < 16; ++cidx) {
                    sum_lo += activation_tile[row * kMmqQ8_1BlockSize + cidx];
                    sum_hi += activation_tile[row * kMmqQ8_1BlockSize + 16 + cidx];
                }
                const float scale = q8_scales[activation_offset] * __half2float(block->d);
                accumulated[index / 32] += scale *
                    (static_cast<float>(block->scales[scale_lo_index]) *
                         (static_cast<float>(dot_lo_tile[warp][index]) - 32.0f * static_cast<float>(sum_lo)) +
                     static_cast<float>(block->scales[scale_lo_index + 1]) *
                         (static_cast<float>(dot_hi_tile[warp][index]) - 32.0f * static_cast<float>(sum_hi)));
            }
        }
        __syncthreads();
    }
    for (int index = lane; index < kMmqTensorCoreTile * kMmqTensorCoreTile; index += 32) {
        const int row = index / kMmqTensorCoreTile;
        const int output = index % kMmqTensorCoreTile;
        if (activation_base + row < m && output_base + output < n) {
            const size_t destination = static_cast<size_t>(activation_base + row) * output_stride + output_base + output;
            float value = accumulated[index / 32];
            if (beta != 0.0f) value += beta * bf16_float(y[destination]);
            y[destination] = __float2bfloat16(value);
        }
    }
}

template <int TileRows>
__global__ void q6k_mmq_prefill_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ q8_scales,
    const float* __restrict__ q8_sums, const uint8_t* __restrict__ blocks,
    __nv_bfloat16* __restrict__ y, int m, int n, int k, size_t row_bytes,
    int output_stride, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    const int activation_base = blockIdx.y * TileRows;
    if (output_row >= n || activation_base >= m) return;

    const int sub_blocks_total = k / kMmqQ8_1BlockSize;
    const uint8_t* row_blocks = blocks + static_cast<size_t>(output_row) * row_bytes;
    float acc[TileRows] = {};
    for (int sub = lane; sub < sub_blocks_total; sub += 32) {
        const int superblock = sub / kQ6KSubBlocksPerSuperBlock;
        const int within = sub % kQ6KSubBlocksPerSuperBlock;
        const BlockQ6K* blk = reinterpret_cast<const BlockQ6K*>(row_blocks) + superblock;
        const int half = within >> 2;
        const int group = within & 3;
        const float d = __half2float(blk->d);
        const float scale_lo = static_cast<float>(blk->scales[half * 8 + group * 2]);
        const float scale_hi = static_cast<float>(blk->scales[half * 8 + group * 2 + 1]);
        int dots_lo[TileRows] = {};
        int dots_hi[TileRows] = {};
        int sums_lo[TileRows] = {};
        int sums_hi[TileRows] = {};
#pragma unroll
        for (int g = 0; g < 16; g += 4) {
            const int packed_w = q6k_pack4(blk, half, group, g);
#pragma unroll
            for (int r = 0; r < TileRows; ++r) if (activation_base + r < m) {
                int packed_a;
                const int8_t* a = q8 + static_cast<size_t>(activation_base + r) * k +
                                  sub * kMmqQ8_1BlockSize + g;
                memcpy(&packed_a, a, sizeof(packed_a));
                dots_lo[r] = __dp4a(packed_w, packed_a, dots_lo[r]);
                sums_lo[r] = __dp4a(0x01010101, packed_a, sums_lo[r]);
            }
        }
#pragma unroll
        for (int g = 16; g < 32; g += 4) {
            const int packed_w = q6k_pack4(blk, half, group, g);
#pragma unroll
            for (int r = 0; r < TileRows; ++r) if (activation_base + r < m) {
                int packed_a;
                const int8_t* a = q8 + static_cast<size_t>(activation_base + r) * k +
                                  sub * kMmqQ8_1BlockSize + g;
                memcpy(&packed_a, a, sizeof(packed_a));
                dots_hi[r] = __dp4a(packed_w, packed_a, dots_hi[r]);
                sums_hi[r] = __dp4a(0x01010101, packed_a, sums_hi[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < TileRows; ++r) if (activation_base + r < m) {
            const size_t activation_offset =
                static_cast<size_t>(activation_base + r) * sub_blocks_total + sub;
            acc[r] += d * q8_scales[activation_offset] *
                (scale_lo * static_cast<float>(dots_lo[r] - 32 * sums_lo[r]) +
                 scale_hi * static_cast<float>(dots_hi[r] - 32 * sums_hi[r]));
        }
    }
#pragma unroll
    for (int r = 0; r < TileRows; ++r) if (activation_base + r < m) {
        const float reduced = warp_sum(acc[r]);
        if (lane == 0) {
            const size_t index = static_cast<size_t>(activation_base + r) * output_stride + output_row;
            float value = reduced;
            if (beta != 0.0f) value += beta * bf16_float(y[index]);
            y[index] = __float2bfloat16(value);
        }
    }
}

} // namespace

CudaDeviceCapabilities discover_cuda_device_capabilities() {
    CudaDeviceCapabilities result;
    if (cudaGetDevice(&result.device_ordinal) != cudaSuccess) return result;
    if (cudaDeviceGetAttribute(&result.compute_major,
                               cudaDevAttrComputeCapabilityMajor,
                               result.device_ordinal) != cudaSuccess) {
        return result;
    }
    if (cudaDeviceGetAttribute(&result.compute_minor,
                               cudaDevAttrComputeCapabilityMinor,
                               result.device_ordinal) != cudaSuccess) {
        return result;
    }
    result.mmq_tensor_core_supported = mmq_tensor_core_supported();
    result.mmq_tensor_core_enabled = mmq_tensor_core_enabled() &&
                                      result.mmq_tensor_core_supported;
    return result;
}

void launch_quantize_q8_1(const __nv_bfloat16* x, int8_t* q8, float* scales,
                          float* sums, int rows, int k, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(k / kMmqQ8_1BlockSize),
                    static_cast<unsigned>(rows));
    quantize_q8_1_kernel<<<grid, kMmqQ8_1BlockSize, 0, stream>>>(
        x, q8, scales, sums, k);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_q4k_mmq_with_policy(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, bool use_tensor_cores,
                    cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    if (m == 1) {
        q4k_mmq_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    } else if (use_tensor_cores && m >= kMmqTensorCoreTile && n >= kMmqTensorCoreTile) {
        constexpr int kTensorWarpsPerBlock = 8;
        constexpr int kTensorColsPerBlock = kMmqTensorCoreTile * kTensorWarpsPerBlock;
        const dim3 tensor_grid(static_cast<unsigned>((n + kTensorColsPerBlock - 1) / kTensorColsPerBlock),
                               static_cast<unsigned>((m + kMmqTensorCoreTile - 1) / kMmqTensorCoreTile));
        q4k_mmq_tensor_core_kernel<kTensorWarpsPerBlock><<<tensor_grid, kTensorWarpsPerBlock * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    } else {
        const dim3 tiled_grid(grid_x, static_cast<unsigned>((m + kMmqPrefillTileRows - 1) /
                                                              kMmqPrefillTileRows));
        q4k_mmq_prefill_kernel<kMmqPrefillTileRows><<<tiled_grid, warps_per_block * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_q4k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream) {
    launch_q4k_mmq_with_policy(q8, q8_scales, q8_sums, blocks, y, m, n, k,
                               row_bytes, output_stride, beta,
                               mmq_tensor_core_enabled(), stream);
}

void launch_q6k_mmq_with_policy(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, bool use_tensor_cores,
                    cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    if (m == 1) {
        q6k_mmq_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    } else if (use_tensor_cores && m >= kMmqTensorCoreTile && n >= kMmqTensorCoreTile) {
        constexpr int kTensorWarpsPerBlock = 8;
        constexpr int kTensorColsPerBlock = kMmqTensorCoreTile * kTensorWarpsPerBlock;
        const dim3 tensor_grid(static_cast<unsigned>((n + kTensorColsPerBlock - 1) / kTensorColsPerBlock),
                               static_cast<unsigned>((m + kMmqTensorCoreTile - 1) / kMmqTensorCoreTile));
        q6k_mmq_tensor_core_kernel<kTensorWarpsPerBlock><<<tensor_grid, kTensorWarpsPerBlock * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    } else {
        const dim3 tiled_grid(grid_x, static_cast<unsigned>((m + kMmqPrefillTileRows - 1) /
                                                              kMmqPrefillTileRows));
        q6k_mmq_prefill_kernel<kMmqPrefillTileRows><<<tiled_grid, warps_per_block * 32, 0, stream>>>(
            q8, q8_scales, q8_sums, blocks, y, m, n, k, row_bytes, output_stride, beta);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_q6k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream) {
    launch_q6k_mmq_with_policy(q8, q8_scales, q8_sums, blocks, y, m, n, k,
                               row_bytes, output_stride, beta,
                               mmq_tensor_core_enabled(), stream);
}

} // namespace celeg
