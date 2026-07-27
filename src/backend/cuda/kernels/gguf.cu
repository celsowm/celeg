#include "lfm/backend/cuda/kernels/gguf.cuh"

#include <cuda_fp16.h>

namespace lfm {
namespace {

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Q4_K super-block (256 elements, 144 bytes):
//   half   d;              // super-block scale for quantized scales
//   half   dmin;           // super-block scale for quantized mins
//   uint8  scales[12];     // 8 x (6-bit scale, 6-bit min), packed
//   uint8  qs[128];        // 256 x 4-bit quants
// Sub-block layout: 8 sub-blocks of 32 elements each.
// Dequant: y = d*sc[j]*q - dmin*m[j], q in [0,15].
// ---------------------------------------------------------------------------
struct BlockQ4K {
    __half d;
    __half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

// Decodes the 6-bit scale/min for sub-block j (0..7) from the packed 12 bytes.
// This matches ggml's get_scale_min_k4.
__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t* q,
                                              uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Returns the dequantized value at logical index `col` (0..255) of one Q4_K
// super-block.
__device__ __forceinline__ float q4k_value(const BlockQ4K* blk, int col) {
    const float d = __half2float(blk->d);
    const float dmin = __half2float(blk->dmin);
    const int sub = col >> 5;          // sub-block 0..7
    const int within = col & 31;       // 0..31
    uint8_t sc, m;
    q4k_scale_min(sub, blk->scales, sc, m);
    // qs is laid out as two nibbles per byte, low nibble for the first 32-wide
    // half of a 64-element group, high nibble for the second.
    const int group = sub >> 1;        // 0..3, each spans 64 elems / 32 bytes
    const uint8_t* qs = blk->qs + group * 32;
    const uint8_t byte = qs[within];
    const int q = (sub & 1) ? (byte >> 4) : (byte & 0xF);
    return d * sc * static_cast<float>(q) - dmin * m;
}

// ---------------------------------------------------------------------------
// Q6_K super-block (256 elements, 210 bytes):
//   uint8  ql[128];        // lower 4 bits
//   uint8  qh[64];         // upper 2 bits
//   int8   scales[16];     // per 16-element sub-block scale
//   half   d;              // super-block scale
// Dequant: y = d * scales[is] * (q - 32), q the 6-bit unsigned value.
// ---------------------------------------------------------------------------
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half d;
};

// Returns the dequantized value at logical index `col` (0..255) of one Q6_K
// super-block. Mirrors ggml's dequantize_row_q6_K layout.
__device__ __forceinline__ float q6k_value(const BlockQ6K* blk, int col) {
    const float d = __half2float(blk->d);
    // The 256 elements are processed in two halves of 128. Within each half,
    // ql spans 64 bytes and qh spans 32 bytes, arranged in 4 groups of 32.
    const int half = col >> 7;          // 0 or 1
    const int idx = col & 127;          // 0..127 inside the half
    const int n = idx & 31;             // position within a 32-wide lane group
    const int grp = idx >> 5;           // 0..3
    const uint8_t* ql = blk->ql + half * 64;
    const uint8_t* qh = blk->qh + half * 32;
    // Reconstruct the 6-bit value following the ggml packing:
    //   is  = 8*half + ... ; the scale sub-block index.
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
    // Scale sub-block index: 16 scales, one per 16 elements. Within a half the
    // scale index is 8*half + (grp*2 + (n >= 16 ? 1 : 0)).
    const int is = half * 8 + grp * 2 + (n >> 4);
    return d * static_cast<float>(blk->scales[is]) * static_cast<float>(q - 32);
}

constexpr int kSuperBlock = 256;

template <typename BlockT, float (*ValueFn)(const BlockT*, int)>
__global__ void gguf_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                                 const uint8_t* __restrict__ blocks,
                                 __nv_bfloat16* __restrict__ y,
                                 int m, int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int blocks_per_row = k / kSuperBlock;
    const size_t row_bytes =
        static_cast<size_t>(blocks_per_row) * sizeof(BlockT);
    const uint8_t* row_blocks =
        blocks + static_cast<size_t>(output_row) * row_bytes;

    for (int activation_row = blockIdx.y; activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;
        float sum = 0.0f;
        // Each lane strides over the 256 elements of every super-block.
        for (int b = 0; b < blocks_per_row; ++b) {
            const BlockT* blk =
                reinterpret_cast<const BlockT*>(row_blocks) + b;
            const int base = b * kSuperBlock;
            for (int c = lane; c < kSuperBlock; c += 32) {
                sum += __bfloat162float(input[base + c]) * ValueFn(blk, c);
            }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            float value = sum;
            const size_t out_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * __bfloat162float(y[out_index]);
            y[out_index] = __float2bfloat16(value);
        }
    }
}

// Dequantizes an entire quantized matrix [n, k] to BF16 (row-major). Used for
// the embedding lookup table so token gather can reuse the BF16 fast path.
// Launched as a flat 1D grid over the n*k elements (row-major).
template <typename BlockT, float (*ValueFn)(const BlockT*, int)>
__global__ void gguf_dequant_kernel(const uint8_t* __restrict__ blocks,
                                    __nv_bfloat16* __restrict__ out,
                                    int n, int k) {
    const size_t total = static_cast<size_t>(n) * k;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int row = static_cast<int>(idx / k);
    const int col = static_cast<int>(idx % k);
    const int blocks_per_row = k / kSuperBlock;
    const size_t row_bytes =
        static_cast<size_t>(blocks_per_row) * sizeof(BlockT);
    const BlockT* row_blocks =
        reinterpret_cast<const BlockT*>(blocks + static_cast<size_t>(row) * row_bytes);
    const BlockT* blk = row_blocks + (col / kSuperBlock);
    const float v = ValueFn(blk, col % kSuperBlock);
    out[idx] = __float2bfloat16(v);
}

} // namespace

void launch_gguf_linear(const __nv_bfloat16* x, const uint8_t* blocks,
                        GgmlType type, __nv_bfloat16* y,
                        int m, int n, int k, float beta,
                        cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid((n + warps_per_block - 1) / warps_per_block, grid_y);
    const int threads = warps_per_block * 32;
    if (type == GgmlType::Q4_K) {
        gguf_gemv_kernel<BlockQ4K, q4k_value>
            <<<grid, threads, 0, stream>>>(x, blocks, y, m, n, k, beta);
    } else if (type == GgmlType::Q6_K) {
        gguf_gemv_kernel<BlockQ6K, q6k_value>
            <<<grid, threads, 0, stream>>>(x, blocks, y, m, n, k, beta);
    }
}

void launch_gguf_dequant(const uint8_t* blocks, GgmlType type,
                         __nv_bfloat16* out, int n, int k,
                         cudaStream_t stream) {
    const dim3 block(256);
    const size_t total = static_cast<size_t>(n) * k;
    const unsigned grid_x =
        static_cast<unsigned>((total + 255) / 256);
    if (type == GgmlType::Q4_K) {
        gguf_dequant_kernel<BlockQ4K, q4k_value>
            <<<grid_x, block, 0, stream>>>(blocks, out, n, k);
    } else if (type == GgmlType::Q6_K) {
        gguf_dequant_kernel<BlockQ6K, q6k_value>
            <<<grid_x, block, 0, stream>>>(blocks, out, n, k);
    }
}

} // namespace lfm
