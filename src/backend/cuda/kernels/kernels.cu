#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/backend/cuda/utils.cuh"

#include <cfloat>
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

__global__ void embedding_value_kernel(int32_t token,
                                       const __nv_bfloat16* table,
                                       __nv_bfloat16* out,
                                       int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        out[i] = table[static_cast<size_t>(token) * hidden + i];
    }
}

__global__ void embedding_pointer_kernel(const int32_t* token,
                                         const __nv_bfloat16* table,
                                         __nv_bfloat16* out,
                                         int hidden) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) {
        out[i] = table[static_cast<size_t>(*token) * hidden + i];
    }
}

__global__ void embedding_batch_kernel(const int32_t* tokens,
                                       int rows,
                                       const __nv_bfloat16* table,
                                       __nv_bfloat16* out,
                                       int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    out[index] = table[static_cast<size_t>(tokens[row]) * hidden + column];
}

__global__ void embedding_int8_value_kernel(int32_t token,
                                             const int8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const float value = static_cast<float>(
        table[static_cast<size_t>(token) * hidden + column]) * scales[token];
    out[column] = __float2bfloat16(value);
}

__global__ void embedding_int8_pointer_kernel(const int32_t* token,
                                               const int8_t* table,
                                               const float* scales,
                                               __nv_bfloat16* out,
                                               int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const int row = *token;
    const float value = static_cast<float>(
        table[static_cast<size_t>(row) * hidden + column]) * scales[row];
    out[column] = __float2bfloat16(value);
}

__global__ void embedding_int8_batch_kernel(const int32_t* tokens,
                                             int rows,
                                             const int8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int batch_row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int token = tokens[batch_row];
    const float value = static_cast<float>(
        table[static_cast<size_t>(token) * hidden + column]) * scales[token];
    out[index] = __float2bfloat16(value);
}

// Weight-only symmetric per-output-channel INT8. Eight warps compute eight
// output rows for one activation row. The decode path (m=1) is therefore a
// true GEMV; prefill reuses the same kernel across activation rows.

__device__ __forceinline__ int unpack_int4(const uint8_t* packed,
                                           int column) {
    const uint8_t byte = packed[column >> 1];
    const uint8_t nibble = (column & 1) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16
                        : static_cast<int>(nibble);
}

__global__ void embedding_int4_value_kernel(int32_t token,
                                             const uint8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(token) * packed_cols;
    out[column] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[token]);
}

__global__ void embedding_int4_pointer_kernel(const int32_t* token,
                                               const uint8_t* table,
                                               const float* scales,
                                               __nv_bfloat16* out,
                                               int hidden) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden) return;
    const int row_index = *token;
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(row_index) * packed_cols;
    out[column] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[row_index]);
}

__global__ void embedding_int4_batch_kernel(const int32_t* tokens,
                                             int rows,
                                             const uint8_t* table,
                                             const float* scales,
                                             __nv_bfloat16* out,
                                             int hidden) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int activation_row = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    const int token = tokens[activation_row];
    const size_t packed_cols = static_cast<size_t>((hidden + 1) / 2);
    const uint8_t* row = table + static_cast<size_t>(token) * packed_cols;
    out[index] = __float2bfloat16(
        static_cast<float>(unpack_int4(row, column)) * scales[token]);
}

// Weight-only symmetric per-output-channel INT4. The packed matrix keeps two
// signed values per byte. Eight warps compute eight output rows per block.
// Shared-memory activation staging: the activation tile is loaded once per
// block and shared across all 8 warps, halving global-memory traffic versus
// the naive per-warp load when m > 1 (multi-row prefill).
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

    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;

        float accum = 0.0f;
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
                float sum = 0.0f;
                for (int column = lane; column < chunk; column += 32) {
                    sum += bf16_float(s_act[column]) *
                           static_cast<float>(unpack_int4(row_weight, base_k + column));
                }
                accum += warp_sum(sum);
            }

            __syncthreads();
        }
        if (output_row < n && lane == 0) {
            float value = accum * row_scale;
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

__global__ void w8a16_linear_kernel(const __nv_bfloat16* x,
                                    const int8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int8_t* row_weight = weight + static_cast<size_t>(output_row) * k;
    // gridDim.y is capped by the CUDA y-dimension limit. Striding keeps the
    // batched path valid for contexts larger than 65,535 tokens.
    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;
        float sum = 0.0f;
        for (int column = lane; column < k; column += 32) {
            sum += bf16_float(input[column]) *
                   static_cast<float>(row_weight[column]);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            float value = sum * scales[output_row];
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

__global__ void rmsnorm_kernel(const __nv_bfloat16* x,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* out,
                               int width,
                               float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* in = x + static_cast<size_t>(row) * width;
    __nv_bfloat16* dst = out + static_cast<size_t>(row) * width;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = bf16_float(in[i]);
        sum += v * v;
    }

    __shared__ float warp_sums[32];
    __shared__ float total;
    sum = block_sum(sum, warp_sums, &total);

    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(width) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(in[i]) * inv);
        dst[i] = __float2bfloat16(normalized * bf16_float(weight[i]));
    }
}

__global__ void residual_kernel(__nv_bfloat16* x,
                                const __nv_bfloat16* residual,
                                int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        x[i] = __float2bfloat16(bf16_float(x[i]) + bf16_float(residual[i]));
    }
}

__global__ void swiglu_fused_kernel(const __nv_bfloat16* gate_up,
                                    __nv_bfloat16* out,
                                    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float gate = bf16_float(gate_up[i]);
        const float up = bf16_float(gate_up[count + i]);
        const float silu = rounded_bf16_float(gate / (1.0f + expf(-gate)));
        out[i] = __float2bfloat16(silu * up);
    }
}

__global__ void conv_decode_kernel(const __nv_bfloat16* bcx,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* state,
                                   __nv_bfloat16* y,
                                   int hidden,
                                   int cache_length,
                                   int position_value,
                                   const int32_t* position_pointer,
                                   bool use_pointer) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;

    const int position = resolved_position(position_value, position_pointer, use_pointer);
    const float b = bf16_float(bcx[channel]);
    const float c = bf16_float(bcx[hidden + channel]);
    const float x = bf16_float(bcx[2 * hidden + channel]);
    const int cursor = position % cache_length;
    state[static_cast<size_t>(cursor) * hidden + channel] =
        __float2bfloat16(b * x);

    float conv = 0.0f;
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int tap = 0; tap < cache_length; ++tap) {
        const int slot = (cursor + 1 + tap) % cache_length;
        conv += bf16_float(state[static_cast<size_t>(slot) * hidden + channel]) *
                bf16_float(weight[weight_offset + tap]);
    }
    const float conv_bf16 = rounded_bf16_float(conv);
    y[channel] = __float2bfloat16(c * conv_bf16);
}

__global__ void conv_prefill_kernel(const __nv_bfloat16* bcx,
                                    const __nv_bfloat16* weight,
                                    __nv_bfloat16* y,
                                    int rows,
                                    int hidden,
                                    int cache_length) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int row = static_cast<int>(index / hidden);
    const int channel = static_cast<int>(index % hidden);
    const size_t stride = static_cast<size_t>(3) * hidden;

    float conv = 0.0f;
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int tap = 0; tap < cache_length; ++tap) {
        const int source_row = row - (cache_length - 1 - tap);
        if (source_row < 0) continue;
        const __nv_bfloat16* source = bcx + static_cast<size_t>(source_row) * stride;
        const float bx = rounded_bf16_float(
            bf16_float(source[channel]) * bf16_float(source[2 * hidden + channel]));
        conv += bx * bf16_float(weight[weight_offset + tap]);
    }

    const __nv_bfloat16* current = bcx + static_cast<size_t>(row) * stride;
    const float conv_bf16 = rounded_bf16_float(conv);
    y[index] = __float2bfloat16(bf16_float(current[hidden + channel]) * conv_bf16);
}

__global__ void conv_prefill_state_kernel(const __nv_bfloat16* bcx,
                                          __nv_bfloat16* state,
                                          int rows,
                                          int hidden,
                                          int cache_length) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;
    const size_t stride = static_cast<size_t>(3) * hidden;
    for (int slot = 0; slot < cache_length; ++slot) {
        state[static_cast<size_t>(slot) * hidden + channel] = __float2bfloat16(0.0f);
    }
    const int start = rows > cache_length ? rows - cache_length : 0;
    for (int row = start; row < rows; ++row) {
        const __nv_bfloat16* source = bcx + static_cast<size_t>(row) * stride;
        const float bx = rounded_bf16_float(
            bf16_float(source[channel]) * bf16_float(source[2 * hidden + channel]));
        state[static_cast<size_t>(row % cache_length) * hidden + channel] =
            __float2bfloat16(bx);
    }
}

__global__ void head_rmsnorm_kernel(__nv_bfloat16* data,
                                    const __nv_bfloat16* norm_weight,
                                    int rows,
                                    int heads,
                                    int head_dim,
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
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(vector[i]) * inv);
        vector[i] = __float2bfloat16(normalized * bf16_float(norm_weight[i]));
    }
}

__global__ void rope_strict_kernel(__nv_bfloat16* data,
                                   const __nv_bfloat16* rope_cos,
                                   const __nv_bfloat16* rope_sin,
                                   int rows,
                                   int heads,
                                   int head_dim,
                                   int position_value,
                                   const int32_t* position_pointer,
                                   int mode) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    const int position = mode == 2 ? row :
        resolved_position(position_value, position_pointer, mode == 1);
    const int half = head_dim / 2;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]);
        const float b = bf16_float(vector[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float s = bf16_float(sin_row[i]);
        const float ac = rounded_bf16_float(a * c);
        const float bs = rounded_bf16_float(b * s);
        const float bc = rounded_bf16_float(b * c);
        const float as = rounded_bf16_float(a * s);
        vector[i] = __float2bfloat16(ac - bs);
        vector[i + half] = __float2bfloat16(bc + as);
    }
}

__global__ void qk_norm_rope_fast_kernel(__nv_bfloat16* data,
                                         const __nv_bfloat16* norm_weight,
                                         const __nv_bfloat16* rope_cos,
                                         const __nv_bfloat16* rope_sin,
                                         int rows,
                                         int heads,
                                         int head_dim,
                                         int position_value,
                                         const int32_t* position_pointer,
                                         int mode,
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
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    const int position = mode == 2 ? row :
        resolved_position(position_value, position_pointer, mode == 1);
    const int half = head_dim / 2;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]);
        const float b = bf16_float(vector[i + half]) * inv *
                        bf16_float(norm_weight[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float s = bf16_float(sin_row[i]);
        vector[i] = __float2bfloat16(a * c - b * s);
        vector[i + half] = __float2bfloat16(b * c + a * s);
    }
}

__global__ void store_kv_kernel(const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                __nv_bfloat16* key_cache,
                                __nv_bfloat16* value_cache,
                                int rows,
                                int width,
                                int position_value,
                                const int32_t* position_pointer,
                                int mode) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    key_cache[static_cast<size_t>(target_row) * width + column] = k[index];
    value_cache[static_cast<size_t>(target_row) * width + column] = v[index];
}

__global__ void store_kv_int8_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_cache, int8_t* value_cache,
    float* key_scales, float* value_scales, int rows, int kv_heads,
    int head_dim, int position_value, const int32_t* position_pointer,
    int mode) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows || head >= kv_heads) return;
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t cache_base =
        (static_cast<size_t>(target_row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index =
            static_cast<size_t>(target_row) * kv_heads + head;
        key_scales[scale_index] = key_scale;
        value_scales[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

__device__ float attention_dot_int8(const __nv_bfloat16* query,
                                    const int8_t* key, float key_scale,
                                    int head_dim, float* warp_sums,
                                    float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) *
            (static_cast<float>(key[d]) * key_scale);
    }
    return block_sum(partial, warp_sums, total);
}

__device__ float attention_dot(const __nv_bfloat16* query,
                               const __nv_bfloat16* key,
                               int head_dim,
                               float* warp_sums,
                               float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) * bf16_float(key[d]);
    }
    return block_sum(partial, warp_sums, total);
}

__global__ void gqa_decode_strict_kernel(const __nv_bfloat16* q,
                                         const __nv_bfloat16* key_cache,
                                         const __nv_bfloat16* value_cache,
                                         __nv_bfloat16* out,
                                         int rows,
                                         int seq_len_value,
                                         const int32_t* position_pointer,
                                         int mode,
                                         int q_heads,
                                         int kv_heads,
                                         int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }

    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator += probability * bf16_float(value[lane]);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_kernel(const __nv_bfloat16* q,
                                         const __nv_bfloat16* key_cache,
                                         const __nv_bfloat16* value_cache,
                                         __nv_bfloat16* out,
                                         int rows,
                                         int seq_len_value,
                                         const int32_t* position_pointer,
                                         int mode,
                                         int q_heads,
                                         int kv_heads,
                                         int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;

    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;

    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();

        if (lane < head_dim) {
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha + bf16_float(value[lane]) * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}


__global__ void gqa_decode_strict_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int seq_len_value, const int32_t* position_pointer, int mode,
    int q_heads, int kv_heads, int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }
    float accumulator = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator += probability *
                (static_cast<float>(value[lane]) * value_scales[scale_index]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int seq_len_value, const int32_t* position_pointer, int mode,
    int q_heads, int kv_heads, int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * value_scales[scale_index] * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }

    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;

    for (int token = begin; token < end; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha +
                bf16_float(value[lane]) * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_partial_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * value_scales[scale_index] * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_reduce_kernel(
    __nv_bfloat16* out, int q_heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_head = blockIdx.x;
    const int lane = threadIdx.x;
    if (query_head >= q_heads) return;
    const size_t base = static_cast<size_t>(query_head) * chunks;
    float global_max = -FLT_MAX;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float local_denom = partial_denom[base + chunk];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + chunk] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            const size_t accum_index =
                (base + chunk) * static_cast<size_t>(head_dim) + lane;
            accumulator += partial_accum[accum_index] * factor;
        }
    }
    if (lane < head_dim) {
        out[static_cast<size_t>(query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void argmax_bf16_kernel(const __nv_bfloat16* values,
                                   int count,
                                   int32_t* result) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];

    float best = -FLT_MAX;
    int index = -1;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = bf16_float(values[i]);
        if (value > best || (value == best && (index < 0 || i < index))) {
            best = value;
            index = i;
        }
    }

    best_values[threadIdx.x] = best;
    best_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const float other_value = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other_value > best_values[threadIdx.x] ||
                (other_value == best_values[threadIdx.x] && other_index >= 0 &&
                 (best_indices[threadIdx.x] < 0 || other_index < best_indices[threadIdx.x]))) {
                best_values[threadIdx.x] = other_value;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = best_indices[0];
}

__global__ void mark_seen_batch_kernel(const int32_t* tokens,
                                       int count,
                                       uint8_t* seen,
                                       int vocab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int token = tokens[i];
        if (token >= 0 && token < vocab) seen[token] = 1;
    }
}

__global__ void mark_seen_batch_ptrs_kernel(const int32_t* tokens,
                                            uint8_t* const* seen,
                                            int rows,
                                            int vocab) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int token = tokens[row];
    if (token >= 0 && token < vocab) seen[row][token] = 1;
}

__global__ void mark_seen_kernel(const int32_t* token,
                                 uint8_t* seen,
                                 int vocab) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int value = *token;
        if (value >= 0 && value < vocab) seen[value] = 1;
    }
}

__global__ void prepare_sampling_scores_kernel(const __nv_bfloat16* logits,
                                               const uint8_t* seen,
                                               float* scores,
                                               int vocab,
                                               float temperature,
                                               float repetition_penalty) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    float value = bf16_float(logits[i]);
    if (seen[i]) {
        value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
    }
    scores[i] = value / temperature;
}

__global__ void select_topk_kernel(float* scores,
                                   float* selected_values,
                                   int32_t* selected_indices,
                                   int rank,
                                   int vocab) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    float best = -FLT_MAX;
    int index = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float value = scores[i];
        if (value > best || (value == best && (index < 0 || i < index))) {
            best = value;
            index = i;
        }
    }
    best_values[threadIdx.x] = best;
    best_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const float other_value = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other_value > best_values[threadIdx.x] ||
                (other_value == best_values[threadIdx.x] && other_index >= 0 &&
                 (best_indices[threadIdx.x] < 0 || other_index < best_indices[threadIdx.x]))) {
                best_values[threadIdx.x] = other_value;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        selected_values[rank] = best_values[0];
        selected_indices[rank] = best_indices[0];
        if (best_indices[0] >= 0) scores[best_indices[0]] = -FLT_MAX;
    }
}

__device__ __forceinline__ uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

__global__ void sample_topk_kernel(const float* selected_values,
                                   const int32_t* selected_indices,
                                   int top_k,
                                   float top_p,
                                   uint64_t* rng_state,
                                   int32_t* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const float maximum = selected_values[0];
    float total = 0.0f;
    for (int i = 0; i < top_k; ++i) total += expf(selected_values[i] - maximum);

    int cutoff = top_k;
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cumulative += expf(selected_values[i] - maximum);
            if (cumulative / total >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    float truncated_total = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        truncated_total += expf(selected_values[i] - maximum);
    }
    uint64_t state = *rng_state;
    const uint64_t random_bits = xorshift64star(state);
    *rng_state = state;
    const double unit = (static_cast<double>(random_bits >> 11) + 0.5) *
                        (1.0 / 9007199254740992.0);
    const float target = static_cast<float>(unit) * truncated_total;
    float cumulative = 0.0f;
    int chosen = cutoff - 1;
    for (int i = 0; i < cutoff; ++i) {
        cumulative += expf(selected_values[i] - maximum);
        if (target <= cumulative) {
            chosen = i;
            break;
        }
    }
    *result = selected_indices[chosen];
}

__global__ void fused_sample_topk_kernel(const __nv_bfloat16* logits,
                                                uint8_t* seen,
                                                float* scores,
                                                float* selected_values,
                                                int32_t* selected_indices,
                                                int vocab,
                                                float temperature,
                                                float repetition_penalty,
                                                int top_k,
                                                float top_p,
                                                uint64_t* rng_state,
                                                int32_t* result) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float value = bf16_float(logits[i]);
        if (seen[i]) {
            value = value < 0.0f ? value * repetition_penalty
                                 : value / repetition_penalty;
        }
        scores[i] = value / temperature;
    }
    __syncthreads();

    for (int rank = 0; rank < top_k; ++rank) {
        float best = -FLT_MAX;
        int index = -1;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            const float value = scores[i];
            if (value > best || (value == best && (index < 0 || i < index))) {
                best = value;
                index = i;
            }
        }
        best_values[threadIdx.x] = best;
        best_indices[threadIdx.x] = index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (static_cast<int>(threadIdx.x) < stride) {
                const float other_value = best_values[threadIdx.x + stride];
                const int other_index = best_indices[threadIdx.x + stride];
                if (other_value > best_values[threadIdx.x] ||
                    (other_value == best_values[threadIdx.x] && other_index >= 0 &&
                     (best_indices[threadIdx.x] < 0 ||
                      other_index < best_indices[threadIdx.x]))) {
                    best_values[threadIdx.x] = other_value;
                    best_indices[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected_values[rank] = best_values[0];
            selected_indices[rank] = best_indices[0];
            if (best_indices[0] >= 0) scores[best_indices[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float maximum = selected_values[0];
        float total = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            total += expf(selected_values[i] - maximum);
        }

        int cutoff = top_k;
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (int i = 0; i < top_k; ++i) {
                cumulative += expf(selected_values[i] - maximum);
                if (cumulative / total >= top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
        }

        float truncated_total = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            truncated_total += expf(selected_values[i] - maximum);
        }
        uint64_t state = *rng_state;
        const uint64_t random_bits = xorshift64star(state);
        *rng_state = state;
        const double unit = (static_cast<double>(random_bits >> 11) + 0.5) *
                            (1.0 / 9007199254740992.0);
        const float target = static_cast<float>(unit) * truncated_total;
        float cumulative = 0.0f;
        int chosen = cutoff - 1;
        for (int i = 0; i < cutoff; ++i) {
            cumulative += expf(selected_values[i] - maximum);
            if (target <= cumulative) {
                chosen = i;
                break;
            }
        }
        *result = selected_indices[chosen];
        const int32_t token = *result;
        if (token >= 0 && token < vocab) seen[token] = 1;
    }
}


__global__ void packed_sample_topk_kernel(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k_values,
    const float* top_p_values,
    float* scores,
    float* selected_values,
    int32_t* selected_indices,
    int rows,
    int vocab,
    int32_t* result) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    const __nv_bfloat16* row_logits = logits[row];
    uint8_t* row_seen = seen[row];
    float* row_scores = scores + static_cast<size_t>(row) * vocab;
    float* row_selected_values =
        selected_values + static_cast<size_t>(row) * 128;
    int32_t* row_selected_indices =
        selected_indices + static_cast<size_t>(row) * 128;
    const float temperature = temperatures[row];
    const float repetition_penalty = repetition_penalties[row];
    const int top_k = top_k_values[row];
    const float top_p = top_p_values[row];
    const bool greedy = temperature <= 0.0f || top_k == 1;

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float value = bf16_float(row_logits[i]);
        if (row_seen[i]) {
            value = value < 0.0f ? value * repetition_penalty
                                 : value / repetition_penalty;
        }
        row_scores[i] = greedy ? value : value / temperature;
    }
    __syncthreads();

    const int ranks = greedy ? 1 : top_k;
    for (int rank = 0; rank < ranks; ++rank) {
        float best = -FLT_MAX;
        int index = -1;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            const float value = row_scores[i];
            if (value > best || (value == best && (index < 0 || i < index))) {
                best = value;
                index = i;
            }
        }
        best_values[threadIdx.x] = best;
        best_indices[threadIdx.x] = index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (static_cast<int>(threadIdx.x) < stride) {
                const float other_value = best_values[threadIdx.x + stride];
                const int other_index = best_indices[threadIdx.x + stride];
                if (other_value > best_values[threadIdx.x] ||
                    (other_value == best_values[threadIdx.x] &&
                     other_index >= 0 &&
                     (best_indices[threadIdx.x] < 0 ||
                      other_index < best_indices[threadIdx.x]))) {
                    best_values[threadIdx.x] = other_value;
                    best_indices[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            row_selected_values[rank] = best_values[0];
            row_selected_indices[rank] = best_indices[0];
            if (best_indices[0] >= 0) row_scores[best_indices[0]] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int chosen_token = row_selected_indices[0];
        if (!greedy) {
            const float maximum = row_selected_values[0];
            float total = 0.0f;
            for (int i = 0; i < top_k; ++i) {
                total += expf(row_selected_values[i] - maximum);
            }
            int cutoff = top_k;
            if (top_p < 1.0f) {
                float cumulative = 0.0f;
                for (int i = 0; i < top_k; ++i) {
                    cumulative += expf(row_selected_values[i] - maximum);
                    if (cumulative / total >= top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }
            }
            float truncated_total = 0.0f;
            for (int i = 0; i < cutoff; ++i) {
                truncated_total += expf(row_selected_values[i] - maximum);
            }
            uint64_t state = *rng_state[row];
            const uint64_t random_bits = xorshift64star(state);
            *rng_state[row] = state;
            const double unit =
                (static_cast<double>(random_bits >> 11) + 0.5) *
                (1.0 / 9007199254740992.0);
            const float target = static_cast<float>(unit) * truncated_total;
            float cumulative = 0.0f;
            int chosen = cutoff - 1;
            for (int i = 0; i < cutoff; ++i) {
                cumulative += expf(row_selected_values[i] - maximum);
                if (target <= cumulative) {
                    chosen = i;
                    break;
                }
            }
            chosen_token = row_selected_indices[chosen];
        }
        result[row] = chosen_token;
        if (chosen_token >= 0 && chosen_token < vocab) {
            row_seen[chosen_token] = 1;
        }
    }
}

__global__ void split_qkv_rows_kernel(const __nv_bfloat16* qkv,
                                      __nv_bfloat16* q,
                                      __nv_bfloat16* k,
                                      __nv_bfloat16* v,
                                      int rows,
                                      int q_width,
                                      int kv_width) {
    const size_t total = static_cast<size_t>(rows) *
                         static_cast<size_t>(q_width + 2 * kv_width);
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int width = q_width + 2 * kv_width;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    if (column < q_width) {
        q[static_cast<size_t>(row) * q_width + column] = qkv[index];
    } else if (column < q_width + kv_width) {
        k[static_cast<size_t>(row) * kv_width + column - q_width] = qkv[index];
    } else {
        v[static_cast<size_t>(row) * kv_width + column - q_width - kv_width] = qkv[index];
    }
}

__global__ void swiglu_interleaved_kernel(const __nv_bfloat16* gate_up,
                                          __nv_bfloat16* out,
                                          int rows,
                                          int intermediate) {
    const size_t total = static_cast<size_t>(rows) * intermediate;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / intermediate);
    const int column = static_cast<int>(index % intermediate);
    const __nv_bfloat16* source = gate_up +
        static_cast<size_t>(row) * 2 * intermediate;
    const float gate = bf16_float(source[column]);
    const float up = bf16_float(source[intermediate + column]);
    const float silu = rounded_bf16_float(gate / (1.0f + expf(-gate)));
    out[index] = __float2bfloat16(silu * up);
}

__global__ void rope_batch_positions_kernel(__nv_bfloat16* data,
                                            const __nv_bfloat16* rope_cos,
                                            const __nv_bfloat16* rope_sin,
                                            const int32_t* positions,
                                            int rows,
                                            int heads,
                                            int head_dim) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    const int position = positions[row];
    const int half = head_dim / 2;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]);
        const float b = bf16_float(vector[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float ss = bf16_float(sin_row[i]);
        const float ac = rounded_bf16_float(a * c);
        const float bs = rounded_bf16_float(b * ss);
        const float bc = rounded_bf16_float(b * c);
        const float as = rounded_bf16_float(a * ss);
        vector[i] = __float2bfloat16(ac - bs);
        vector[i + half] = __float2bfloat16(bc + as);
    }
}

__global__ void qk_norm_rope_fast_batch_positions_kernel(
    __nv_bfloat16* data,
    const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions,
    int rows,
    int heads,
    int head_dim,
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
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();
    const int position = positions[row];
    const int half = head_dim / 2;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]);
        const float b = bf16_float(vector[i + half]) * inv *
                        bf16_float(norm_weight[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float ss = bf16_float(sin_row[i]);
        vector[i] = __float2bfloat16(a * c - b * ss);
        vector[i + half] = __float2bfloat16(b * c + a * ss);
    }
}

__global__ void store_kv_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width) {
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const size_t target = static_cast<size_t>(positions[row]) * kv_width + column;
    key_cache[row][target] = k[index];
    value_cache[row][target] = v[index];
}

__global__ void store_kv_int8_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t scale_index =
        static_cast<size_t>(positions[row]) * kv_heads + head;
    const size_t cache_base = scale_index * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        key_scales[row][scale_index] = key_scale;
        value_scales[row][scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

__global__ void gqa_decode_online_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const __nv_bfloat16* row_keys = key_cache[row];
    const __nv_bfloat16* row_values = value_cache[row];
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = row_values +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha + bf16_float(value[lane]) * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_strict_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const __nv_bfloat16* row_keys = key_cache[row];
    const __nv_bfloat16* row_values = value_cache[row];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }
    float accumulator = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = row_values +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator += probability * bf16_float(value[lane]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_int8_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const int8_t* row_keys = key_cache[row];
    const int8_t* row_values = value_cache[row];
    const float* row_key_scales = key_scales[row];
    const float* row_value_scales = value_scales[row];
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = row_values + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * row_value_scales[scale_index] * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_strict_int8_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const int8_t* row_keys = key_cache[row];
    const int8_t* row_values = value_cache[row];
    const float* row_key_scales = key_scales[row];
    const float* row_value_scales = value_scales[row];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }
    float accumulator = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = row_values + scale_index * head_dim;
            accumulator += probability *
                (static_cast<float>(value[lane]) * row_value_scales[scale_index]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void conv_decode_batch_ptrs_kernel(
    const __nv_bfloat16* bcx,
    const __nv_bfloat16* weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    int rows,
    int hidden,
    int cache_length) {
    const int row = blockIdx.y;
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || channel >= hidden) return;
    const __nv_bfloat16* source = bcx +
        static_cast<size_t>(row) * 3 * hidden;
    __nv_bfloat16* state = states[row];
    const int position = positions[row];
    const float b = bf16_float(source[channel]);
    const float c = bf16_float(source[hidden + channel]);
    const float x = bf16_float(source[2 * hidden + channel]);
    const int cursor = position % cache_length;
    state[static_cast<size_t>(cursor) * hidden + channel] =
        __float2bfloat16(b * x);
    float conv = 0.0f;
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int tap = 0; tap < cache_length; ++tap) {
        const int slot = (cursor + 1 + tap) % cache_length;
        conv += bf16_float(state[static_cast<size_t>(slot) * hidden + channel]) *
                bf16_float(weight[weight_offset + tap]);
    }
    y[static_cast<size_t>(row) * hidden + channel] =
        __float2bfloat16(c * rounded_bf16_float(conv));
}

__global__ void scatter_bf16_rows_kernel(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    destinations[row][column] = source[index];
}

__global__ void scatter_decode_state_kernel(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    *sampled_destinations[row] = sampled[row];
    *position_destinations[row] = positions[row] + 1;
}

__global__ void increment_position_kernel(int32_t* position) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ++(*position);
}

int attention_threads(int head_dim) {
    int threads = 32;
    while (threads < head_dim && threads < 1024) threads <<= 1;
    return threads;
}

void launch_qk_common(__nv_bfloat16* q,
                      __nv_bfloat16* k,
                      const __nv_bfloat16* q_norm,
                      const __nv_bfloat16* k_norm,
                      const __nv_bfloat16* rope_cos,
                      const __nv_bfloat16* rope_sin,
                      int rows,
                      int q_heads,
                      int kv_heads,
                      int head_dim,
                      int position_value,
                      const int32_t* position_pointer,
                      int mode,
                      float eps,
                      bool fast,
                      cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        qk_norm_rope_fast_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rope_cos, rope_sin, rows, q_heads, head_dim,
            position_value, position_pointer, mode, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, rows, q_heads, head_dim,
            position_value, position_pointer, mode);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode);
        LFM_KERNEL_DEBUG_SYNC(stream);
    }
}

} // namespace

void launch_embedding(int32_t token, const __nv_bfloat16* table,
                      __nv_bfloat16* out, int hidden, cudaStream_t stream) {
    embedding_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_device(const int32_t* token, const __nv_bfloat16* table,
                             __nv_bfloat16* out, int hidden, cudaStream_t stream) {
    embedding_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_batch(const int32_t* tokens, int rows,
                            const __nv_bfloat16* table, __nv_bfloat16* out,
                            int hidden, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8(int32_t token, const int8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) {
    embedding_int8_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8_device(const int32_t* token, const int8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream) {
    embedding_int8_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int8_batch(const int32_t* tokens, int rows,
                                 const int8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_int8_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_w8a16_linear(const __nv_bfloat16* x, const int8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid((n + warps_per_block - 1) / warps_per_block, grid_y);
    w8a16_linear_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        x, weight, scales, y, m, n, k, beta);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_embedding_int4(int32_t token, const uint8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) {
    embedding_int4_value_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int4_device(const int32_t* token, const uint8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream) {
    embedding_int4_pointer_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        token, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_embedding_int4_batch(const int32_t* tokens, int rows,
                                 const uint8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    embedding_int4_batch_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        tokens, rows, table, scales, out, hidden);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_w4a16_linear(const __nv_bfloat16* x, const uint8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    // Use 512-element tiles (1 KB smem). For k <= 512, one chunk; for larger k,
    // multiple chunks sharing the same smem buffer per activation row.
    constexpr int tile_k = 512;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    const size_t smem_size = static_cast<size_t>(tile_k) * sizeof(__nv_bfloat16);
    w4a16_linear_kernel<<<grid, warps_per_block * 32, smem_size, stream>>>(
        x, weight, scales, y, m, n, k, beta, tile_k);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    __nv_bfloat16* out, int rows, int width, float eps,
                    cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, width, eps);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                         int count, cudaStream_t stream) {
    residual_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, residual, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                         int count, cudaStream_t stream) {
    swiglu_fused_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_decode(const __nv_bfloat16* projected_bcx,
                        const __nv_bfloat16* conv_weight,
                        __nv_bfloat16* state, __nv_bfloat16* y,
                        int hidden, int cache_length, int position,
                        cudaStream_t stream) {
    conv_decode_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, conv_weight, state, y, hidden, cache_length,
        position, nullptr, false);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_decode_device(const __nv_bfloat16* projected_bcx,
                               const __nv_bfloat16* conv_weight,
                               __nv_bfloat16* state, __nv_bfloat16* y,
                               int hidden, int cache_length,
                               const int32_t* position,
                               cudaStream_t stream) {
    conv_decode_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, conv_weight, state, y, hidden, cache_length,
        0, position, true);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_prefill(const __nv_bfloat16* projected_bcx,
                         const __nv_bfloat16* conv_weight,
                         __nv_bfloat16* state, __nv_bfloat16* y,
                         int rows, int hidden, int cache_length,
                         cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    conv_prefill_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        projected_bcx, conv_weight, y, rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
    conv_prefill_state_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, state, rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_qk_norm_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                const __nv_bfloat16* q_norm,
                                const __nv_bfloat16* k_norm,
                                const __nv_bfloat16* rope_cos,
                                const __nv_bfloat16* rope_sin,
                                int q_heads, int kv_heads, int head_dim,
                                int position, float eps,
                                cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, position, nullptr,
                     0, eps, false, stream);
}

void launch_qk_norm_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                       const __nv_bfloat16* q_norm,
                                       const __nv_bfloat16* k_norm,
                                       const __nv_bfloat16* rope_cos,
                                       const __nv_bfloat16* rope_sin,
                                       int q_heads, int kv_heads, int head_dim,
                                       const int32_t* position, float eps,
                                       cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, 0, position,
                     1, eps, false, stream);
}

void launch_qk_norm_rope_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                              const __nv_bfloat16* q_norm,
                              const __nv_bfloat16* k_norm,
                              const __nv_bfloat16* rope_cos,
                              const __nv_bfloat16* rope_sin,
                              int q_heads, int kv_heads, int head_dim,
                              int position, float eps,
                              cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, position, nullptr,
                     0, eps, true, stream);
}

void launch_qk_norm_rope_fast_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                     const __nv_bfloat16* q_norm,
                                     const __nv_bfloat16* k_norm,
                                     const __nv_bfloat16* rope_cos,
                                     const __nv_bfloat16* rope_sin,
                                     int q_heads, int kv_heads, int head_dim,
                                     const int32_t* position, float eps,
                                     cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, 0, position,
                     1, eps, true, stream);
}

void launch_qk_norm_rope_prefill_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                        const __nv_bfloat16* q_norm,
                                        const __nv_bfloat16* k_norm,
                                        const __nv_bfloat16* rope_cos,
                                        const __nv_bfloat16* rope_sin,
                                        int rows, int q_heads, int kv_heads,
                                        int head_dim, float eps,
                                        cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     rows, q_heads, kv_heads, head_dim, 0, nullptr,
                     2, eps, false, stream);
}

void launch_qk_norm_rope_prefill_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                                      const __nv_bfloat16* q_norm,
                                      const __nv_bfloat16* k_norm,
                                      const __nv_bfloat16* rope_cos,
                                      const __nv_bfloat16* rope_sin,
                                      int rows, int q_heads, int kv_heads,
                                      int head_dim, float eps,
                                      cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     rows, q_heads, kv_heads, head_dim, 0, nullptr,
                     2, eps, true, stream);
}

void launch_store_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                     __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
                     int position, int kv_width, cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, position, nullptr, 0);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_device(const __nv_bfloat16* k, const __nv_bfloat16* v,
                            __nv_bfloat16* key_cache,
                            __nv_bfloat16* value_cache,
                            const int32_t* position, int kv_width,
                            cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, 0, position, 1);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_prefill(const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* key_cache,
                             __nv_bfloat16* value_cache,
                             int rows, int kv_width, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, rows, kv_width, 0, nullptr, 2);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int8_t* key_cache, int8_t* value_cache,
                          float* key_scales, float* value_scales,
                          int position, int kv_heads, int head_dim,
                          cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, position, nullptr, 0);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_device(const __nv_bfloat16* k,
                                 const __nv_bfloat16* v,
                                 int8_t* key_cache, int8_t* value_cache,
                                 float* key_scales, float* value_scales,
                                 const int32_t* position, int kv_heads,
                                 int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, 0, position, 1);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_prefill(const __nv_bfloat16* k,
                                  const __nv_bfloat16* v,
                                  int8_t* key_cache, int8_t* value_cache,
                                  float* key_scales, float* value_scales,
                                  int rows, int kv_heads, int head_dim,
                                  cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, rows,
        kv_heads, head_dim, 0, nullptr, 2);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, seq_len, nullptr, 0,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, 0, position, 1,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, seq_len, nullptr, 0,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, 0, position, 1,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_segmented_device(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_segment_partial_kernel<<<q_heads * chunks, threads, 0, stream>>>(
        q, key_cache, value_cache, position, q_heads, kv_heads, head_dim,
        chunk_tokens, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_strict(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, rows, 0, nullptr, 2,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_online(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, rows, 0, nullptr, 2,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        seq_len, nullptr, 0, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        seq_len, nullptr, 0, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        0, position, 1, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        0, position, 1, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_segmented_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_segment_partial_int8_kernel<<<q_heads * chunks, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, position,
        q_heads, kv_heads, head_dim, chunk_tokens, chunks, partial_max,
        partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, rows,
        0, nullptr, 2, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, rows,
        0, nullptr, 2, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_argmax_bf16(const __nv_bfloat16* logits, int count,
                        int32_t* result, cudaStream_t stream) {
    argmax_bf16_kernel<<<1, 256, 0, stream>>>(logits, count, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch(const int32_t* tokens, int count,
                            uint8_t* seen, int vocab, cudaStream_t stream) {
    mark_seen_batch_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        tokens, count, seen, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch_ptrs(const int32_t* tokens,
                                 uint8_t* const* seen,
                                 int rows, int vocab,
                                 cudaStream_t stream) {
    mark_seen_batch_ptrs_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        tokens, seen, rows, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen(const int32_t* token, uint8_t* seen, int vocab,
                      cudaStream_t stream) {
    mark_seen_kernel<<<1, 1, 0, stream>>>(token, seen, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_prepare_sampling_scores(const __nv_bfloat16* logits,
                                    const uint8_t* seen,
                                    float* scores, int vocab,
                                    float temperature,
                                    float repetition_penalty,
                                    cudaStream_t stream) {
    prepare_sampling_scores_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
        logits, seen, scores, vocab, temperature, repetition_penalty);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_select_topk(float* scores, float* selected_values,
                        int32_t* selected_indices, int rank, int vocab,
                        cudaStream_t stream) {
    select_topk_kernel<<<1, 256, 0, stream>>>(
        scores, selected_values, selected_indices, rank, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_sample_topk(const float* selected_values,
                        const int32_t* selected_indices,
                        int top_k, float top_p, uint64_t* rng_state,
                        int32_t* result, cudaStream_t stream) {
    sample_topk_kernel<<<1, 1, 0, stream>>>(
        selected_values, selected_indices, top_k, top_p, rng_state, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_fused_sample_topk(const __nv_bfloat16* logits,
                              uint8_t* seen,
                              float* scores,
                              float* selected_values,
                              int32_t* selected_indices,
                              int vocab,
                              float temperature,
                              float repetition_penalty,
                              int top_k,
                              float top_p,
                              uint64_t* rng_state,
                              int32_t* result,
                              cudaStream_t stream) {
    fused_sample_topk_kernel<<<1, 256, 0, stream>>>(
        logits, seen, scores, selected_values, selected_indices, vocab,
        temperature, repetition_penalty, top_k, top_p, rng_state, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_packed_sample_topk(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k,
    const float* top_p,
    float* scores,
    float* selected_values,
    int32_t* selected_indices,
    int rows,
    int vocab,
    int32_t* result,
    cudaStream_t stream) {
    packed_sample_topk_kernel<<<rows, 256, 0, stream>>>(
        logits, seen, rng_state, temperatures, repetition_penalties,
        top_k, top_p, scores, selected_values, selected_indices,
        rows, vocab, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_split_qkv_rows(const __nv_bfloat16* qkv,
                           __nv_bfloat16* q,
                           __nv_bfloat16* k,
                           __nv_bfloat16* v,
                           int rows,
                           int q_width,
                           int kv_width,
                           cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) *
                         static_cast<size_t>(q_width + 2 * kv_width);
    split_qkv_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        qkv, q, k, v, rows, q_width, kv_width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_interleaved(const __nv_bfloat16* gate_up,
                               __nv_bfloat16* out,
                               int rows,
                               int intermediate,
                               cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * intermediate;
    swiglu_interleaved_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        gate_up, out, rows, intermediate);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_qk_norm_rope_batch_positions(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* q_norm,
    const __nv_bfloat16* k_norm,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    float eps,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        qk_norm_rope_fast_batch_positions_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rope_cos, rope_sin, positions, rows,
            q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, positions, rows,
            kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, positions, rows, q_heads, head_dim);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, positions, rows, kv_heads, head_dim);
        LFM_KERNEL_DEBUG_SYNC(stream);
    }
}


__device__ __forceinline__ size_t paged_vector_offset(
    uint32_t page, int attention_slot, int in_page, int head, int dim,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    return (((((static_cast<size_t>(page) * attention_layers + attention_slot) *
               page_tokens + in_page) * kv_heads + head) * head_dim) + dim);
}

__device__ __forceinline__ size_t paged_scale_offset(
    uint32_t page, int attention_slot, int in_page, int head,
    int page_tokens, int attention_layers, int kv_heads) {
    return ((((static_cast<size_t>(page) * attention_layers + attention_slot) *
              page_tokens + in_page) * kv_heads) + head);
}

__global__ void store_kv_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    const size_t kv_width = static_cast<size_t>(kv_heads) * head_dim;
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const int head = column / head_dim;
    const int dim = column % head_dim;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t target = paged_vector_offset(page, attention_slot, in_page,
                                               head, dim, page_tokens,
                                               attention_layers, kv_heads,
                                               head_dim);
    key_pool[target] = k[index];
    value_pool[target] = v[index];
}

__global__ void store_kv_int8_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index = paged_scale_offset(
            page, attention_slot, in_page, head, page_tokens,
            attention_layers, kv_heads);
        key_scale_pool[scale_index] = key_scale;
        value_scale_pool[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const size_t target = paged_vector_offset(
            page, attention_slot, in_page, head, d, page_tokens,
            attention_layers, kv_heads, head_dim);
        key_pool[target] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_pool[target] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

template <bool Strict>
__global__ void gqa_decode_paged_batch_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if constexpr (Strict) {
        if (lane == 0) maximum = -FLT_MAX;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                maximum = fmaxf(maximum, score);
            }
            __syncthreads();
        }
        if (lane == 0) denominator = 0.0f;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                denominator += expf(score - maximum);
            }
            __syncthreads();
        }
        float accumulator = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                probability = rounded_bf16_float(expf(score - maximum) / denominator);
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator += probability * bf16_float(value_pool[offset]);
            }
            __syncthreads();
        }
        if (lane < head_dim) {
            out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
                __float2bfloat16(accumulator);
        }
    } else {
        float running_max = -FLT_MAX;
        float denom = 0.0f;
        float accumulator = 0.0f;
        __shared__ float alpha;
        __shared__ float beta;
        __shared__ float next_max;
        __shared__ float next_denom;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = dot * scale;
                next_max = fmaxf(running_max, score);
                alpha = expf(running_max - next_max);
                beta = expf(score - next_max);
                next_denom = denom * alpha + beta;
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator = accumulator * alpha + bf16_float(value_pool[offset]) * beta;
            }
            denom = next_denom;
            running_max = next_max;
            __syncthreads();
        }
        if (lane < head_dim) {
            out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
                __float2bfloat16(accumulator / denom);
        }
    }
}


__device__ float paged_int8_attention_dot(
    const __nv_bfloat16* query,
    const int8_t* key_pool,
    const float* key_scale_pool,
    uint32_t page, int attention_slot, int in_page, int kv_head,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    float* warp_sums, float* dot_total) {
    const int lane = threadIdx.x;
    const size_t scale_offset = paged_scale_offset(
        page, attention_slot, in_page, kv_head, page_tokens,
        attention_layers, kv_heads);
    const float key_scale = key_scale_pool[scale_offset];
    float local = 0.0f;
    for (int d = lane; d < head_dim; d += blockDim.x) {
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, d, page_tokens,
            attention_layers, kv_heads, head_dim);
        local += bf16_float(query[d]) * static_cast<float>(key_pool[offset]) * key_scale;
    }
    return block_sum(local, warp_sums, dot_total);
}

template <bool Strict>
__global__ void gqa_decode_int8_paged_batch_kernel(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if constexpr (Strict) {
        if (lane == 0) maximum = -FLT_MAX;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) maximum = fmaxf(maximum,
                rounded_bf16_float(rounded_bf16_float(dot) * scale));
            __syncthreads();
        }
        if (lane == 0) denominator = 0.0f;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) denominator += expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum);
            __syncthreads();
        }
        float accumulator = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) probability = rounded_bf16_float(expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum) / denominator);
            __syncthreads();
            if (lane < head_dim) {
                const size_t scale_offset = paged_scale_offset(
                    page, attention_slot, ip, kv_head, page_tokens,
                    attention_layers, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator += probability * static_cast<float>(value_pool[offset]) *
                               value_scale_pool[scale_offset];
            }
            __syncthreads();
        }
        if (lane < head_dim) out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    } else {
        float running_max = -FLT_MAX, denom = 0.0f, accumulator = 0.0f;
        __shared__ float alpha, beta, next_max, next_denom;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) {
                const float score = dot * scale;
                next_max = fmaxf(running_max, score);
                alpha = expf(running_max - next_max);
                beta = expf(score - next_max);
                next_denom = denom * alpha + beta;
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t scale_offset = paged_scale_offset(
                    page, attention_slot, ip, kv_head, page_tokens,
                    attention_layers, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator = accumulator * alpha + static_cast<float>(value_pool[offset]) *
                              value_scale_pool[scale_offset] * beta;
            }
            denom = next_denom;
            running_max = next_max;
            __syncthreads();
        }
        if (lane < head_dim) out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denom);
    }
}


__global__ void gqa_decode_paged_segment_partial_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const int lp = token / page_tokens;
        const int ip = token % page_tokens;
        const uint32_t page = page_tables[
            static_cast<size_t>(row) * page_table_stride + lp];
        float local = 0.0f;
        for (int d = lane; d < head_dim; d += blockDim.x) {
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, d, page_tokens,
                attention_layers, kv_heads, head_dim);
            local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
        }
        const float dot = block_sum(local, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, lane, page_tokens,
                attention_layers, kv_heads, head_dim);
            accumulator = accumulator * alpha +
                bf16_float(value_pool[offset]) * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_int8_paged_segment_partial_kernel(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const int lp = token / page_tokens;
        const int ip = token % page_tokens;
        const uint32_t page = page_tables[
            static_cast<size_t>(row) * page_table_stride + lp];
        const float dot = paged_int8_attention_dot(
            query, key_pool, key_scale_pool, page, attention_slot, ip,
            kv_head, page_tokens, attention_layers, kv_heads, head_dim,
            warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const size_t scale_index = paged_scale_offset(
                page, attention_slot, ip, kv_head, page_tokens,
                attention_layers, kv_heads);
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, lane, page_tokens,
                attention_layers, kv_heads, head_dim);
            accumulator = accumulator * alpha +
                static_cast<float>(value_pool[offset]) *
                value_scale_pool[scale_index] * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_reduce_batch_kernel(
    __nv_bfloat16* out, int rows, int q_heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_index = blockIdx.x;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    const size_t base =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks;
    float global_max = -FLT_MAX;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float local_denom = partial_denom[base + chunk];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + chunk] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            accumulator += partial_accum[
                (base + chunk) * static_cast<size_t>(head_dim) + lane] * factor;
        }
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

void launch_store_kv_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_batch_ptrs_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, positions, rows, kv_width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_batch_ptrs_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales,
        positions, rows, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_batch_ptrs(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        gqa_decode_online_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, out, positions,
            rows, q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_strict_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, out, positions,
            rows, q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_int8_batch_ptrs(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        gqa_decode_online_int8_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, key_scales, value_scales,
            out, positions, rows, q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_strict_int8_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, key_scales, value_scales,
            out, positions, rows, q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_decode_batch_ptrs(
    const __nv_bfloat16* projected_bcx,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    int rows,
    int hidden,
    int cache_length,
    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((hidden + 255) / 256),
                    static_cast<unsigned>(rows));
    conv_decode_batch_ptrs_kernel<<<grid, 256, 0, stream>>>(
        projected_bcx, conv_weight, states, y, positions,
        rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_bf16_rows(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    scatter_bf16_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        source, destinations, rows, width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_decode_state(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows,
    cudaStream_t stream) {
    scatter_decode_state_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sampled, positions, sampled_destinations,
        position_destinations, rows);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_increment_position(int32_t* position, cudaStream_t stream) {
    increment_position_kernel<<<1, 1, 0, stream>>>(position);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_store_kv_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * kv_heads * head_dim;
    store_kv_paged_batch_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        k, v, key_pool, value_pool, page_tables, page_table_stride,
        positions, rows, attention_slot, page_tokens, attention_layers,
        kv_heads, head_dim);
    LFM_KERNEL_CHECK();
}

void launch_store_kv_int8_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream) {
    store_kv_int8_paged_batch_kernel<<<rows * kv_heads, 64, 0, stream>>>(
        k, v, key_pool, value_pool, key_scale_pool, value_scale_pool,
        page_tables, page_table_stride, positions, rows, attention_slot,
        page_tokens, attention_layers, kv_heads, head_dim);
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_paged_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream) {
    if (fast) {
        gqa_decode_paged_batch_kernel<false><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, page_tables, page_table_stride, out,
            positions, rows, attention_slot, page_tokens, attention_layers,
            q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_paged_batch_kernel<true><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, page_tables, page_table_stride, out,
            positions, rows, attention_slot, page_tokens, attention_layers,
            q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream) {
    if (fast) {
        gqa_decode_int8_paged_batch_kernel<false><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, key_scale_pool, value_scale_pool,
            page_tables, page_table_stride, out, positions, rows,
            attention_slot, page_tokens, attention_layers, q_heads,
            kv_heads, head_dim);
    } else {
        gqa_decode_int8_paged_batch_kernel<true><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, key_scale_pool, value_scale_pool,
            page_tables, page_table_stride, out, positions, rows,
            attention_slot, page_tokens, attention_layers, q_heads,
            kv_heads, head_dim);
    }
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_paged_segmented_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_paged_segment_partial_kernel<<<rows * q_heads * chunks, threads, 0, stream>>>(
        q, key_pool, value_pool, page_tables, page_table_stride, positions,
        rows, attention_slot, page_tokens, attention_layers, q_heads,
        kv_heads, head_dim, chunk_tokens, chunks, partial_max,
        partial_denom, partial_accum);
    LFM_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_segmented_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_int8_paged_segment_partial_kernel<<<rows * q_heads * chunks, threads, 0, stream>>>(
        q, key_pool, value_pool, key_scale_pool, value_scale_pool,
        page_tables, page_table_stride, positions, rows, attention_slot,
        page_tokens, attention_layers, q_heads, kv_heads, head_dim,
        chunk_tokens, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_CHECK();
}

} // namespace lfm
