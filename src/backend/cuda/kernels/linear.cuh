#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

__global__ void w4a16_linear_kernel(const __nv_bfloat16* x,
                                    const uint8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta,
                                    int tile_k) {
    constexpr int warps_per_block = 8;
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_act = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block) return;

    const size_t packed_cols = static_cast<size_t>((k + 1) / 2);
    const uint8_t* row_weight = (output_row < n)
        ? (weight + static_cast<size_t>(output_row) * packed_cols)
        : nullptr;
    const float row_scale = (output_row < n) ? scales[output_row] : 0.0f;
    const bool row_weight_aligned =
        (output_row < n) && ((reinterpret_cast<uintptr_t>(row_weight) & 3u) == 0);

    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;

        float accum_partial = 0.0f;
        for (int base_k = 0; base_k < k; base_k += tile_k) {
            const int chunk = (k - base_k) < tile_k ? (k - base_k) : tile_k;

            {
                const int tid = threadIdx.x;
                const int stride = blockDim.x;
                for (int i = tid; i < chunk; i += stride) {
                    s_act[i] = input[base_k + i];
                }
            }
            __syncthreads();

            if (output_row < n) {
                int column = 0;
                if (row_weight_aligned) {
                    const uint32_t* words =
                        reinterpret_cast<const uint32_t*>(row_weight + (base_k >> 1));
                    for (int group = lane * 8; group + 8 <= chunk; group += 32 * 8) {
                        const uint32_t word = words[group >> 3];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const uint8_t nibble = (word >> (i * 4)) & 0xFU;
                            const int value = nibble >= 8U
                                ? static_cast<int>(nibble) - 16
                                : static_cast<int>(nibble);
                            accum_partial += bf16_float(s_act[group + i]) *
                                              static_cast<float>(value);
                        }
                    }
                    column = (chunk / 8) * 8;
                }
                for (int c = column + lane; c < chunk; c += 32) {
                    accum_partial += bf16_float(s_act[c]) *
                        static_cast<float>(unpack_int4(row_weight, base_k + c));
                }
            }

            __syncthreads();
        }
        if (output_row < n) {
            const float accum = warp_sum(accum_partial);
            if (lane == 0) {
                float value = accum * row_scale;
                const size_t output_index =
                    static_cast<size_t>(activation_row) * n + output_row;
                if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
                y[output_index] = __float2bfloat16(value);
            }
        }
    }
}

void launch_w8a16_linear(const __nv_bfloat16* x, const int8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid((n + warps_per_block - 1) / warps_per_block, grid_y);
    w8a16_gemv_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        x, weight, scales, y, m, n, k, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}


void launch_w4a16_linear(const __nv_bfloat16* x, const uint8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    constexpr int tile_k = 512;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    const size_t smem_size = static_cast<size_t>(tile_k) * sizeof(__nv_bfloat16);
    w4a16_linear_kernel<<<grid, warps_per_block * 32, smem_size, stream>>>(
        x, weight, scales, y, m, n, k, beta, tile_k);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

/// Dynamic per-row FP8 E4M3 quantization: used both for per-token activation
/// quantization at inference time and (identically, just run once at
/// conversion time rather than per forward pass) for per-channel static
/// weight quantization -- both are "row absmax -> e4m3" in the same layout,
/// so one kernel serves both roles.
__global__ void quantize_e4m3_per_row_kernel(const __nv_bfloat16* __restrict__ x,
                                             __nv_fp8_e4m3* __restrict__ q,
                                             float* __restrict__ scales,
                                             int k) {
    const int row = blockIdx.x;
    const __nv_bfloat16* row_x = x + static_cast<size_t>(row) * k;
    __nv_fp8_e4m3* row_q = q + static_cast<size_t>(row) * k;

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(bf16_float(row_x[i])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    __shared__ float warp_max[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_max[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        const int warps = (blockDim.x + 31) / 32;
        float v = (lane < warps) ? warp_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
        }
        if (lane == 0) warp_max[0] = v;
    }
    __syncthreads();
    const float row_max = warp_max[0];

    // e4m3's largest finite magnitude (448) as the quantization ceiling.
    constexpr float kFp8E4m3Max = 448.0f;
    const float scale = row_max > 0.0f ? row_max / kFp8E4m3Max : 1.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        row_q[i] = __nv_fp8_e4m3(bf16_float(row_x[i]) * inv_scale);
    }
}

void launch_quantize_e4m3_per_row(const __nv_bfloat16* x, __nv_fp8_e4m3* q,
                                  float* scales, int rows, int k,
                                  cudaStream_t stream) {
    constexpr int threads = 256;
    quantize_e4m3_per_row_kernel<<<rows, threads, 0, stream>>>(x, q, scales, k);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

/// Applies the W8A8 dequant scale to a raw (unscaled) FP8xFP8->FP32 matmul
/// accumulation. cuBLASLt's native per-channel/per-token scale-vector mode
/// (CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F) is not supported on this
/// codebase's target hardware (RTX 5090 / CUDA 13.2 -- confirmed empirically,
/// see docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3), so the outer-product
/// scale (act_scale[row] * weight_scale[col]) is applied here instead, the
/// same "scale multiply inside a hand-written epilogue" convention
/// launch_w8a16_linear already uses for the W8A16 path.
__global__ void fp8_scale_apply_kernel(const float* __restrict__ raw,
                                       const float* __restrict__ act_scale,
                                       const float* __restrict__ weight_scale,
                                       __nv_bfloat16* __restrict__ y,
                                       int m, int n, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n || row >= m) return;
    const size_t idx = static_cast<size_t>(row) * n + col;
    float value = raw[idx] * act_scale[row] * weight_scale[col];
    if (beta != 0.0f) value += beta * bf16_float(y[idx]);
    y[idx] = __float2bfloat16(value);
}

void launch_fp8_scale_apply(const float* raw, const float* act_scale,
                            const float* weight_scale, __nv_bfloat16* y,
                            int m, int n, float beta, cudaStream_t stream) {
    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned>((n + threads - 1) / threads),
                    static_cast<unsigned>(m));
    fp8_scale_apply_kernel<<<grid, threads, 0, stream>>>(
        raw, act_scale, weight_scale, y, m, n, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Naive fallback matmul for shapes the fp8 cuBLASLt heuristic can't produce
// an algorithm for (e.g. n not a multiple of the tensor-core tile size --
// see docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 3). One thread per output
// element, straight-line dot product. Not fast, but every shape is valid, so
// GemmDispatcher can fall back to it instead of throwing.
__global__ void fp8_w8a8_naive_kernel(const __nv_fp8_e4m3* __restrict__ x_q,
                                      const float* __restrict__ act_scale,
                                      const __nv_fp8_e4m3* __restrict__ w_q,
                                      const float* __restrict__ weight_scale,
                                      __nv_bfloat16* __restrict__ y,
                                      int m, int n, int k, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n || row >= m) return;
    const __nv_fp8_e4m3* x_row = x_q + static_cast<size_t>(row) * k;
    const __nv_fp8_e4m3* w_row = w_q + static_cast<size_t>(col) * k;
    float acc = 0.0f;
    for (int i = 0; i < k; ++i) {
        acc += float(x_row[i]) * float(w_row[i]);
    }
    float value = acc * act_scale[row] * weight_scale[col];
    const size_t idx = static_cast<size_t>(row) * n + col;
    if (beta != 0.0f) value += beta * bf16_float(y[idx]);
    y[idx] = __float2bfloat16(value);
}

void launch_fp8_w8a8_naive(const __nv_fp8_e4m3* x_q, const float* act_scale,
                           const __nv_fp8_e4m3* w_q, const float* weight_scale,
                           __nv_bfloat16* y, int m, int n, int k, float beta,
                           cudaStream_t stream) {
    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned>((n + threads - 1) / threads),
                    static_cast<unsigned>(m));
    fp8_w8a8_naive_kernel<<<grid, threads, 0, stream>>>(
        x_q, act_scale, w_q, weight_scale, y, m, n, k, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Dequantizes a packed NVFP4 (e2m1, 2 values/byte) weight with per-16-block
// UE4M3 scales + a per-tensor fp32 global scale into bf16, so it can run
// through the existing bf16 GEMM path. FALLBACK ONLY -- see
// docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 4: the primary path is now
// the native cuBLASLt VEC16_UE4M3 block-scaled fp4 matmul (see
// quantize_e2m1_per_block_kernel / swizzle_nvfp4_scale_128x4_kernel below),
// once the correct scale-factor swizzle (NVIDIA's documented 128x4 tiled
// layout) was found. This dequant path is kept only for shapes where
// get_or_create_nvfp4_lt_plan can't find an algorithm, mirroring
// launch_fp8_w8a8_naive's role for the fp8 path.
__global__ void dequant_nvfp4_kernel(const uint8_t* __restrict__ packed,
                                     const __nv_fp8_e4m3* __restrict__ block_scales,
                                     float global_scale,
                                     __nv_bfloat16* __restrict__ out,
                                     int rows, int cols, int block_size) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= cols || row >= rows) return;
    const int blocks_per_row = cols / block_size;
    const int block = col / block_size;
    const size_t elem = static_cast<size_t>(row) * cols + col;
    const uint8_t byte = packed[elem / 2];
    const uint8_t nibble = (col & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0xF);
    const __half_raw h = __nv_cvt_fp4_to_halfraw(
        static_cast<__nv_fp4_storage_t>(nibble), __NV_E2M1);
    const float scale = float(block_scales[static_cast<size_t>(row) * blocks_per_row + block]);
    const float value = __half2float(__half(h)) * scale * global_scale;
    out[elem] = __float2bfloat16(value);
}

void launch_dequant_nvfp4(const uint8_t* packed, const __nv_fp8_e4m3* block_scales,
                          float global_scale, __nv_bfloat16* out,
                          int rows, int cols, int block_size, cudaStream_t stream) {
    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned>((cols + threads - 1) / threads),
                    static_cast<unsigned>(rows));
    dequant_nvfp4_kernel<<<grid, threads, 0, stream>>>(
        packed, block_scales, global_scale, out, rows, cols, block_size);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Quantizes bf16 to packed NVFP4 (e2m1, 2/byte) with a per-16-block UE4M3
// scale (row-major [rows, cols/block_size]), given a per-tensor fp32
// global_scale calibration factor (dequant = e2m1 * block_scale *
// global_scale, matching the two-level compressed-tensors NVFP4 scheme).
// One thread per 16-element block -- blocks never overlap, so each thread
// owns its 8 packed output bytes exclusively.
__global__ void quantize_e2m1_per_block_kernel(const __nv_bfloat16* __restrict__ x,
                                               uint8_t* __restrict__ packed,
                                               __nv_fp8_e4m3* __restrict__ scales,
                                               int rows, int cols, int block_size,
                                               float global_scale) {
    const int blocks_per_row = cols / block_size;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_blocks = rows * blocks_per_row;
    if (idx >= total_blocks) return;
    const int row = idx / blocks_per_row;
    const int block = idx % blocks_per_row;
    const size_t base = static_cast<size_t>(row) * cols + static_cast<size_t>(block) * block_size;

    float absmax = 0.0f;
    for (int i = 0; i < block_size; ++i) {
        absmax = fmaxf(absmax, fabsf(bf16_float(x[base + i])));
    }
    constexpr float kE2m1Max = 6.0f;
    const float raw_scale = absmax > 0.0f ? absmax / (kE2m1Max * global_scale) : 1.0f;
    const __nv_fp8_e4m3 quant_scale(raw_scale);
    scales[static_cast<size_t>(row) * blocks_per_row + block] = quant_scale;
    const float inv_eff_scale = 1.0f / (float(quant_scale) * global_scale);

    for (int i = 0; i < block_size; i += 2) {
        const uint8_t lo = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
            bf16_float(x[base + i]) * inv_eff_scale, __NV_E2M1, cudaRoundNearest));
        const uint8_t hi = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
            bf16_float(x[base + i + 1]) * inv_eff_scale, __NV_E2M1, cudaRoundNearest));
        packed[(base + i) / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

void launch_quantize_e2m1_per_block(const __nv_bfloat16* x, uint8_t* packed,
                                    __nv_fp8_e4m3* scales, int rows, int cols,
                                    int block_size, float global_scale,
                                    cudaStream_t stream) {
    const int blocks_per_row = cols / block_size;
    const int total_blocks = rows * blocks_per_row;
    constexpr int threads = 128;
    const int blocks_grid = (total_blocks + threads - 1) / threads;
    quantize_e2m1_per_block_kernel<<<blocks_grid, threads, 0, stream>>>(
        x, packed, scales, rows, cols, block_size, global_scale);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Rearranges a row-major [rows, k_scale] UE4M3 scale tensor into the 128x4
// tiled layout cuBLASLt's CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 mode
// expects: NVIDIA's documented layout (128 rows x 4 scale-columns per
// 512-byte tile, offset = (row%32)*16 + (row/32)*4 + col within a tile,
// tiles row-major, rows padded to 128 / scale-columns padded to 4 with
// zero-fill) -- see docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 4. `dst`
// must already be zeroed (padding relies on it) and sized
// tiles_m*tiles_n*512 where tiles_m=ceil(rows/128), tiles_n=ceil(k_scale/4).
__global__ void swizzle_nvfp4_scale_kernel(const __nv_fp8_e4m3* __restrict__ src,
                                           __nv_fp8_e4m3* __restrict__ dst,
                                           int rows, int k_scale, int tiles_n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= k_scale || row >= rows) return;
    const int tile_r = row / 128, tile_c = col / 4;
    const int outer = row % 128, inner = col % 4;
    const int tile_idx = tile_r * tiles_n + tile_c;
    const int local = (outer % 32) * 16 + (outer / 32) * 4 + inner;
    dst[static_cast<size_t>(tile_idx) * 512 + local] = src[static_cast<size_t>(row) * k_scale + col];
}

void launch_swizzle_nvfp4_scale(const __nv_fp8_e4m3* src, __nv_fp8_e4m3* dst,
                                int rows, int k_scale, cudaStream_t stream) {
    const int tiles_n = (k_scale + 3) / 4;
    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned>((k_scale + threads - 1) / threads),
                    static_cast<unsigned>(rows));
    swizzle_nvfp4_scale_kernel<<<grid, threads, 0, stream>>>(src, dst, rows, k_scale, tiles_n);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

// Applies the two per-tensor NVFP4 global scales (weight's and
// activation's) that CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 doesn't apply
// itself -- it only bakes in the per-16-block UE4M3 scale, so the raw
// matmul output is off by weight_global_scale * act_global_scale.
__global__ void nvfp4_global_scale_apply_kernel(const float* __restrict__ raw,
                                                float total_scale,
                                                __nv_bfloat16* __restrict__ y,
                                                int m, int n, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= n || row >= m) return;
    const size_t idx = static_cast<size_t>(row) * n + col;
    float value = raw[idx] * total_scale;
    if (beta != 0.0f) value += beta * bf16_float(y[idx]);
    y[idx] = __float2bfloat16(value);
}

void launch_nvfp4_global_scale_apply(const float* raw, float total_scale,
                                     __nv_bfloat16* y, int m, int n, float beta,
                                     cudaStream_t stream) {
    constexpr int threads = 128;
    const dim3 grid(static_cast<unsigned>((n + threads - 1) / threads),
                    static_cast<unsigned>(m));
    nvfp4_global_scale_apply_kernel<<<grid, threads, 0, stream>>>(raw, total_scale, y, m, n, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
