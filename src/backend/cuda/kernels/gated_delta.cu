#include "kernel_common.cuh"
#include "celeg/backend/cuda/kernels/gated_delta.hpp"

#include <algorithm>
#include <cmath>

namespace celeg {
namespace {

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ float softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

__global__ void interleave_gated_delta_qkv_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* qkv, int rows, int key_width, int value_width) {
    const size_t total = static_cast<size_t>(rows) *
        static_cast<size_t>(2 * key_width + value_width);
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int row = static_cast<int>(index / static_cast<size_t>(2 * key_width + value_width));
        const int column = static_cast<int>(index % static_cast<size_t>(2 * key_width + value_width));
        const size_t source_row = static_cast<size_t>(row);
        if (column < key_width) {
            qkv[index] = q[source_row * key_width + column];
        } else if (column < 2 * key_width) {
            qkv[index] = k[source_row * key_width + (column - key_width)];
        } else {
            qkv[index] = v[source_row * value_width + (column - 2 * key_width)];
        }
    }
}

// For long, contiguous prefill sequences, time remains sequential but each
// state value column is independent.  Prepare q/k in one ordered block per
// key head, execute independent value-column tiles in parallel, then normalize
// each value head in an ordered block.  Key/value grouping is expressed by the
// neutral topology: every value head reads its owning key head, while each
// convolution channel remains uniquely owned by this prepare block.
__global__ void gated_delta_sequence_prepare_kernel(
    __nv_bfloat16* qkv, const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state, int rows, int conv_kernel, int key_head_dim,
    int value_head_dim, int key_heads, int value_heads, float eps) {
    const int key_head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (key_head >= key_heads) return;
    const int repeat = value_heads / key_heads;
    const int key_width = key_heads * key_head_dim;
    const int value_width = value_heads * value_head_dim;
    const int qkv_width = 2 * key_width + value_width;
    __shared__ float reduce[16];
    for (int row = 0; row < rows; ++row) {
        __nv_bfloat16* current = qkv + static_cast<size_t>(row) * qkv_width;
        auto convolve = [&](int channel) {
            __nv_bfloat16* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
            for (int tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
            history[conv_kernel - 1] = current[channel];
            float filtered = 0.0f;
            for (int tap = 0; tap < conv_kernel; ++tap) {
                filtered += bf16_float(history[tap]) * bf16_float(
                    conv_weight[static_cast<size_t>(channel) * conv_kernel + tap]);
            }
            return filtered * sigmoid(filtered);
        };
        float q = 0.0f, k = 0.0f;
        if (lane < key_head_dim) {
            const int offset = key_head * key_head_dim + lane;
            q = convolve(offset); k = convolve(key_width + offset);
            current[offset] = __float2bfloat16(q);
            current[key_width + offset] = __float2bfloat16(k);
        }
        for (int value_offset = lane; value_offset < repeat * value_head_dim;
             value_offset += blockDim.x) {
            const int offset = 2 * key_width + key_head * repeat * value_head_dim + value_offset;
            current[offset] = __float2bfloat16(convolve(offset));
        }
        float q2 = q * q, k2 = k * k;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q2 += __shfl_down_sync(0xffffffffu, q2, offset);
            k2 += __shfl_down_sync(0xffffffffu, k2, offset);
        }
        if ((lane & 31) == 0) { reduce[lane >> 5] = q2; reduce[8 + (lane >> 5)] = k2; }
        __syncthreads();
        if (lane == 0) {
            float q_total = 0.0f, k_total = 0.0f;
            for (int warp = 0; warp < (blockDim.x + 31) / 32; ++warp) {
                q_total += reduce[warp]; k_total += reduce[8 + warp];
            }
            reduce[0] = rsqrtf(q_total + eps); reduce[1] = rsqrtf(k_total + eps);
        }
        __syncthreads();
        if (lane < key_head_dim) {
            const int offset = key_head * key_head_dim + lane;
            current[offset] = __float2bfloat16(bf16_float(current[offset]) * reduce[0]);
            current[key_width + offset] = __float2bfloat16(
                bf16_float(current[key_width + offset]) * reduce[1]);
        }
        __syncthreads();
    }
}

template <int StaticKeyHeadDim = 0>
__global__ void gated_delta_sequence_state_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* b, const __nv_bfloat16* a,
    const __nv_bfloat16* dt_bias, const __nv_bfloat16* a_log,
    __nv_bfloat16* recurrent_state, __nv_bfloat16* output, int rows,
    int key_head_dim, int value_head_dim, int key_heads, int value_heads, bool vector_decay,
    bool safe_decay, float decay_lower_bound) {
    constexpr int columns_per_block = 4;
    const int value_head = static_cast<int>(blockIdx.x);
    const int v_dim = static_cast<int>(blockIdx.y) * columns_per_block + threadIdx.y;
    const int lane = threadIdx.x;
    if (value_head >= value_heads || v_dim >= value_head_dim) return;
    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const int head_dim = StaticKeyHeadDim == 0 ? key_head_dim : StaticKeyHeadDim;
    const int key_width = key_heads * head_dim;
    const int value_width = value_heads * value_head_dim;
    const int qkv_width = 2 * key_width + value_width;
    __nv_bfloat16* state = recurrent_state + static_cast<size_t>(value_head) * head_dim * value_head_dim;
    float state_registers[StaticKeyHeadDim == 0 ? 8 : StaticKeyHeadDim / 32];
    constexpr int register_shards = StaticKeyHeadDim == 0 ? 8 : StaticKeyHeadDim / 32;
    const int shards = (head_dim + 31) / 32;
#pragma unroll
    for (int shard = 0; shard < register_shards; ++shard) {
        const int k_dim = shard * 32 + lane;
        if (shard < shards && k_dim < head_dim) {
            state_registers[shard] = bf16_float(
                state[static_cast<size_t>(k_dim) * value_head_dim + v_dim]);
        }
    }
    for (int row = 0; row < rows; ++row) {
        const __nv_bfloat16* current = qkv + static_cast<size_t>(row) * qkv_width;
        const __nv_bfloat16* row_a = a + static_cast<size_t>(row) *
            (vector_decay ? key_width : value_heads);
        float scalar_decay = 0.0f;
        if (!vector_decay) {
            float warp_decay = 0.0f;
            if (lane == 0) {
                const float raw_decay = bf16_float(row_a[value_head]) +
                    bf16_float(dt_bias[value_head]);
                warp_decay = safe_decay
                    ? expf(sigmoid(expf(bf16_float(a_log[value_head])) * raw_decay) *
                        decay_lower_bound)
                    : expf(-expf(bf16_float(a_log[value_head])) * softplus(raw_decay));
            }
            scalar_decay = __shfl_sync(0xffffffffu, warp_decay, 0);
        }
        float memory_partial = 0.0f;
#pragma unroll
        for (int shard = 0; shard < register_shards; ++shard) {
            const int k_dim = shard * 32 + lane;
            if (shard >= shards || k_dim >= head_dim) continue;
            const int decay_index = key_head * head_dim + k_dim;
            const float decay = vector_decay
                ? (safe_decay
                    ? expf(sigmoid(expf(bf16_float(a_log[value_head])) *
                        (bf16_float(row_a[decay_index]) + bf16_float(dt_bias[decay_index]))) * decay_lower_bound)
                    : expf(-expf(bf16_float(a_log[value_head])) *
                        softplus(bf16_float(row_a[decay_index]) + bf16_float(dt_bias[decay_index]))))
                : scalar_decay;
            state_registers[shard] = bf16_float(__float2bfloat16(
                decay * state_registers[shard]));
            memory_partial += state_registers[shard] *
                bf16_float(current[key_width + key_head * head_dim + k_dim]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            memory_partial += __shfl_down_sync(0xffffffffu, memory_partial, offset);
        }
        const float memory = __shfl_sync(0xffffffffu, memory_partial, 0);
        const float beta = sigmoid(bf16_float(b[static_cast<size_t>(row) * value_heads + value_head]));
        const float value = bf16_float(current[2 * key_width + value_head * value_head_dim + v_dim]);
        const float delta = (value - memory) * beta;
        float result_partial = 0.0f;
#pragma unroll
        for (int shard = 0; shard < register_shards; ++shard) {
            const int k_dim = shard * 32 + lane;
            if (shard >= shards || k_dim >= head_dim) continue;
            state_registers[shard] = bf16_float(__float2bfloat16(state_registers[shard] +
                bf16_float(current[key_width + key_head * head_dim + k_dim]) * delta));
            result_partial += state_registers[shard] *
                bf16_float(current[key_head * head_dim + k_dim]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            result_partial += __shfl_down_sync(0xffffffffu, result_partial, offset);
        }
        if (lane == 0) output[static_cast<size_t>(row) * value_width + value_head * value_head_dim + v_dim] =
            __float2bfloat16(result_partial / sqrtf(static_cast<float>(head_dim)));
    }

#pragma unroll
    for (int shard = 0; shard < register_shards; ++shard) {
        const int k_dim = shard * 32 + lane;
        if (shard < shards && k_dim < head_dim) {
            state[static_cast<size_t>(k_dim) * value_head_dim + v_dim] =
                __float2bfloat16(state_registers[shard]);
        }
    }
}

__global__ void gated_delta_sequence_norm_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* z, const __nv_bfloat16* norm_weight,
    int rows, int value_head_dim, int value_heads, float eps, bool sigmoid_output_gate) {
    const int value_head = static_cast<int>(blockIdx.x), lane = static_cast<int>(threadIdx.x);
    if (value_head >= value_heads) return;
    const int value_width = value_heads * value_head_dim;
    __shared__ float partial[128];
    for (int row = 0; row < rows; ++row) {
        __nv_bfloat16* values = output + static_cast<size_t>(row) * value_width + value_head * value_head_dim;
        float sum = 0.0f;
        for (int d = lane; d < value_head_dim; d += blockDim.x) { const float value = bf16_float(values[d]); sum += value * value; }
        partial[lane] = sum; __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) { if (lane < offset) partial[lane] += partial[lane + offset]; __syncthreads(); }
        const float inverse = rsqrtf(partial[0] / value_head_dim + eps);
        for (int d = lane; d < value_head_dim; d += blockDim.x) {
            const float gate = bf16_float(z[static_cast<size_t>(row) * value_width + value_head * value_head_dim + d]);
            values[d] = __float2bfloat16(bf16_float(values[d]) * inverse * bf16_float(norm_weight[d]) *
                (sigmoid_output_gate ? sigmoid(gate) : gate * sigmoid(gate)));
        }
        __syncthreads();
    }
}

__global__ void gated_delta_net_kernel(
    const __nv_bfloat16* projected_qkv, const __nv_bfloat16* projected_z,
    const __nv_bfloat16* projected_b, const __nv_bfloat16* projected_a,
    const __nv_bfloat16* conv_weight, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* a_log, const __nv_bfloat16* norm_weight,
    __nv_bfloat16* conv_state, __nv_bfloat16* recurrent_state,
    __nv_bfloat16* output, int rows, int conv_kernel, int key_head_dim,
    int value_head_dim, int key_heads, int value_heads, float eps,
    bool vector_decay, bool safe_decay, float decay_lower_bound,
    bool sigmoid_output_gate) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int key_width = key_heads * key_head_dim;
    const int value_width = value_heads * value_head_dim;
    const int qkv_width = 2 * key_width + value_width;
    const int repeat = value_heads / key_heads;

    for (int row = 0; row < rows; ++row) {
        __nv_bfloat16* qkv = const_cast<__nv_bfloat16*>(projected_qkv) +
            static_cast<size_t>(row) * qkv_width;
        const __nv_bfloat16* z = projected_z + static_cast<size_t>(row) * value_width;
        const __nv_bfloat16* b = projected_b + static_cast<size_t>(row) * value_heads;
        const __nv_bfloat16* a = projected_a + static_cast<size_t>(row) * value_heads;

        for (int channel = 0; channel < qkv_width; ++channel) {
            __nv_bfloat16* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
            for (int tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
            history[conv_kernel - 1] = qkv[channel];
            float filtered = 0.0f;
            for (int tap = 0; tap < conv_kernel; ++tap) {
                filtered += bf16_float(history[tap]) * bf16_float(
                    conv_weight[static_cast<size_t>(channel) * conv_kernel + tap]);
            }
            qkv[channel] = __float2bfloat16(filtered * sigmoid(filtered));
        }

        for (int head = 0; head < key_heads; ++head) {
            float q_norm = 0.0f;
            float k_norm = 0.0f;
            for (int d = 0; d < key_head_dim; ++d) {
                const float q = bf16_float(qkv[head * key_head_dim + d]);
                const float k = bf16_float(qkv[key_width + head * key_head_dim + d]);
                q_norm += q * q;
                k_norm += k * k;
            }
            q_norm = sqrtf(q_norm + eps);
            k_norm = sqrtf(k_norm + eps);
            for (int d = 0; d < key_head_dim; ++d) {
                qkv[head * key_head_dim + d] = __float2bfloat16(
                    bf16_float(qkv[head * key_head_dim + d]) / q_norm);
                qkv[key_width + head * key_head_dim + d] = __float2bfloat16(
                    bf16_float(qkv[key_width + head * key_head_dim + d]) / k_norm);
            }
        }

        for (int value_head = 0; value_head < value_heads; ++value_head) {
            const int key_head = value_head / repeat;
            const float beta = sigmoid(bf16_float(b[value_head]));
            __nv_bfloat16* state = recurrent_state + static_cast<size_t>(value_head) *
                key_head_dim * value_head_dim;
            for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                const int decay_index = vector_decay ? key_head * key_head_dim + k_dim
                                                      : value_head;
                const float decay = safe_decay
                    ? expf(sigmoid(expf(bf16_float(a_log[value_head])) *
                        (bf16_float(a[decay_index]) + bf16_float(dt_bias[decay_index]))) *
                        decay_lower_bound)
                    : expf(-expf(bf16_float(a_log[value_head])) *
                        softplus(bf16_float(a[decay_index]) + bf16_float(dt_bias[decay_index])));
                for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
                    const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + v_dim;
                    state[offset] = __float2bfloat16(decay * bf16_float(state[offset]));
                }
            }
            for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
                float memory = 0.0f;
                for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                    memory += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + v_dim]) *
                        bf16_float(qkv[key_width + key_head * key_head_dim + k_dim]);
                }
                const float delta = (bf16_float(qkv[2 * key_width +
                    value_head * value_head_dim + v_dim]) - memory) * beta;
                for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                    const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + v_dim;
                    state[offset] = __float2bfloat16(bf16_float(state[offset]) +
                        bf16_float(qkv[key_width + key_head * key_head_dim + k_dim]) * delta);
                }
            }
            for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
                float value = 0.0f;
                for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                    value += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + v_dim]) *
                        bf16_float(qkv[key_head * key_head_dim + k_dim]);
                }
                output[static_cast<size_t>(row) * value_width +
                       value_head * value_head_dim + v_dim] = __float2bfloat16(
                    value / sqrtf(static_cast<float>(key_head_dim)));
            }
        }

        for (int value_head = 0; value_head < value_heads; ++value_head) {
            float sum = 0.0f;
            for (int d = 0; d < value_head_dim; ++d) {
                const float value = bf16_float(output[static_cast<size_t>(row) * value_width +
                    value_head * value_head_dim + d]);
                sum += value * value;
            }
            const float inv = rsqrtf(sum / value_head_dim + eps);
            for (int d = 0; d < value_head_dim; ++d) {
                const size_t offset = static_cast<size_t>(row) * value_width +
                    value_head * value_head_dim + d;
                const float gate = bf16_float(z[value_head * value_head_dim + d]);
                output[offset] = __float2bfloat16(
                    bf16_float(output[offset]) * inv * bf16_float(norm_weight[d]) *
                    (sigmoid_output_gate ? sigmoid(gate) : gate * sigmoid(gate)));
            }
        }
    }
}

// Time is sequential for a recurrent token, but channels and heads within a
// token are independent.  Keep the scalar kernel above as the general
// multi-row fallback; use these parallel stages for the single-token decode
// path.
__global__ void gated_delta_conv_kernel(
    __nv_bfloat16* qkv, const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state, int qkv_width, int conv_kernel) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= qkv_width) return;
    __nv_bfloat16* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
    for (int tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
    history[conv_kernel - 1] = qkv[channel];
    float filtered = 0.0f;
    for (int tap = 0; tap < conv_kernel; ++tap) {
        filtered += bf16_float(history[tap]) * bf16_float(
            conv_weight[static_cast<size_t>(channel) * conv_kernel + tap]);
    }
    qkv[channel] = __float2bfloat16(filtered * sigmoid(filtered));
}

__global__ void gated_delta_qk_norm_kernel(
    __nv_bfloat16* qkv, int key_head_dim, int key_heads, int key_width,
    float eps) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= key_heads || threadIdx.x != 0) return;
    float q_norm = 0.0f, k_norm = 0.0f;
    for (int d = 0; d < key_head_dim; ++d) {
        const float q = bf16_float(qkv[head * key_head_dim + d]);
        const float k = bf16_float(qkv[key_width + head * key_head_dim + d]);
        q_norm += q * q; k_norm += k * k;
    }
    q_norm = sqrtf(q_norm + eps); k_norm = sqrtf(k_norm + eps);
    for (int d = 0; d < key_head_dim; ++d) {
        qkv[head * key_head_dim + d] = __float2bfloat16(
            bf16_float(qkv[head * key_head_dim + d]) / q_norm);
        qkv[key_width + head * key_head_dim + d] = __float2bfloat16(
            bf16_float(qkv[key_width + head * key_head_dim + d]) / k_norm);
    }
}

__global__ void gated_delta_recurrent_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* z,
    const __nv_bfloat16* b, const __nv_bfloat16* a,
    const __nv_bfloat16* dt_bias, const __nv_bfloat16* a_log,
    const __nv_bfloat16* norm_weight, __nv_bfloat16* recurrent_state,
    __nv_bfloat16* output, int key_head_dim, int value_head_dim,
    int key_heads, int value_heads, float eps, bool vector_decay,
    bool safe_decay, float decay_lower_bound, bool sigmoid_output_gate) {
    const int value_head = static_cast<int>(blockIdx.x);
    const int v_dim = static_cast<int>(threadIdx.x);
    if (value_head >= value_heads) return;
    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const int key_width = key_heads * key_head_dim;
    extern __shared__ float shared[];
    float* decay = shared;
    float* inverse = shared + key_head_dim;
    if (v_dim < key_head_dim) {
        const int decay_index = vector_decay ? key_head * key_head_dim + v_dim : value_head;
        decay[v_dim] = safe_decay
            ? expf(sigmoid(expf(bf16_float(a_log[value_head])) *
                (bf16_float(a[decay_index]) + bf16_float(dt_bias[decay_index]))) * decay_lower_bound)
            : expf(-expf(bf16_float(a_log[value_head])) *
                softplus(bf16_float(a[decay_index]) + bf16_float(dt_bias[decay_index])));
    }
    __syncthreads();
    __nv_bfloat16* state = recurrent_state + static_cast<size_t>(value_head) *
        key_head_dim * value_head_dim;
    if (v_dim < value_head_dim) {
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + v_dim;
            state[offset] = __float2bfloat16(decay[k_dim] * bf16_float(state[offset]));
        }
        float memory = 0.0f;
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            memory += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + v_dim]) *
                bf16_float(qkv[key_width + key_head * key_head_dim + k_dim]);
        }
        const float delta = (bf16_float(qkv[2 * key_width + value_head * value_head_dim + v_dim]) -
            memory) * sigmoid(bf16_float(b[value_head]));
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + v_dim;
            state[offset] = __float2bfloat16(bf16_float(state[offset]) +
                bf16_float(qkv[key_width + key_head * key_head_dim + k_dim]) * delta);
        }
        float value = 0.0f;
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            value += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + v_dim]) *
                bf16_float(qkv[key_head * key_head_dim + k_dim]);
        }
        output[value_head * value_head_dim + v_dim] = __float2bfloat16(
            value / sqrtf(static_cast<float>(key_head_dim)));
    }
    __syncthreads();
    if (v_dim == 0) {
        float sum = 0.0f;
        for (int d = 0; d < value_head_dim; ++d) {
            const float value = bf16_float(output[value_head * value_head_dim + d]);
            sum += value * value;
        }
        *inverse = rsqrtf(sum / value_head_dim + eps);
    }
    __syncthreads();
    if (v_dim < value_head_dim) {
        const int offset = value_head * value_head_dim + v_dim;
        const float gate = bf16_float(z[offset]);
        output[offset] = __float2bfloat16(bf16_float(output[offset]) * *inverse *
            bf16_float(norm_weight[v_dim]) *
            (sigmoid_output_gate ? sigmoid(gate) : gate * sigmoid(gate)));
    }
}

// A token's recurrent state evolves sequentially, but a one-to-one key/value
// head topology has no cross-block data dependency.  Fuse convolution, q/k
// normalization, state update, and output normalization into that one block.
// This removes two launches for each recurrent layer without attaching the
// implementation to a checkpoint identity.  Grouped-query layouts retain the
// generic path below because more than one value head would otherwise update
// the same key-channel convolution history.
__global__ void gated_delta_fused_single_head_kernel(
    const __nv_bfloat16* projected_qkv, const __nv_bfloat16* z,
    const __nv_bfloat16* b, const __nv_bfloat16* a,
    const __nv_bfloat16* conv_weight, const __nv_bfloat16* dt_bias,
    const __nv_bfloat16* a_log, const __nv_bfloat16* norm_weight,
    __nv_bfloat16* conv_state, __nv_bfloat16* recurrent_state,
    __nv_bfloat16* output, int rows, int conv_kernel, int key_head_dim,
    int value_head_dim, int key_heads, float eps, bool vector_decay,
    bool safe_decay, float decay_lower_bound, bool sigmoid_output_gate) {
    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (head >= key_heads) return;

    const int key_width = key_heads * key_head_dim;
    const int qkv_width = 2 * key_width + key_heads * value_head_dim;
    const int value_width = key_heads * value_head_dim;
    __nv_bfloat16* qkv = nullptr;
    extern __shared__ float shared[];
    float* q_values = shared;
    float* k_values = q_values + key_head_dim;
    float* decay = k_values + key_head_dim;
    float* reductions = decay + key_head_dim;
    float* inverse = reductions + 8;

    auto convolve_channel = [&](int channel) {
        __nv_bfloat16* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
        const __nv_bfloat16* weight = conv_weight + static_cast<size_t>(channel) * conv_kernel;
        if (conv_kernel == 4) {
            history[0] = history[1]; history[1] = history[2];
            history[2] = history[3]; history[3] = qkv[channel];
            const float filtered = bf16_float(history[0]) * bf16_float(weight[0]) +
                bf16_float(history[1]) * bf16_float(weight[1]) +
                bf16_float(history[2]) * bf16_float(weight[2]) +
                bf16_float(history[3]) * bf16_float(weight[3]);
            return filtered * sigmoid(filtered);
        }
        for (int tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
        history[conv_kernel - 1] = qkv[channel];
        float filtered = 0.0f;
        for (int tap = 0; tap < conv_kernel; ++tap) {
            filtered += bf16_float(history[tap]) * bf16_float(
                weight[tap]);
        }
        return filtered * sigmoid(filtered);
    };

    for (int row = 0; row < rows; ++row) {
    qkv = const_cast<__nv_bfloat16*>(projected_qkv) + static_cast<size_t>(row) * qkv_width;
    const __nv_bfloat16* z_row = z + static_cast<size_t>(row) * value_width;
    const __nv_bfloat16* b_row = b + static_cast<size_t>(row) * key_heads;
    const __nv_bfloat16* a_row = a + static_cast<size_t>(row) *
        (vector_decay ? key_width : key_heads);
    __nv_bfloat16* output_row = output + static_cast<size_t>(row) * value_width;
    float v_value = 0.0f;
    if (lane < key_head_dim) {
        q_values[lane] = convolve_channel(head * key_head_dim + lane);
        k_values[lane] = convolve_channel(key_width + head * key_head_dim + lane);
    }
    if (lane < value_head_dim) {
        v_value = convolve_channel(2 * key_width + head * value_head_dim + lane);
    }
    __syncthreads();

    float q_square = 0.0f;
    float k_square = 0.0f;
    for (int d = lane; d < key_head_dim; d += blockDim.x) {
        q_square += q_values[d] * q_values[d];
        k_square += k_values[d] * k_values[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_square += __shfl_down_sync(0xffffffffu, q_square, offset);
        k_square += __shfl_down_sync(0xffffffffu, k_square, offset);
    }
    if ((lane & 31) == 0) {
        reductions[lane >> 5] = q_square;
        reductions[4 + (lane >> 5)] = k_square;
    }
    __syncthreads();
    if (lane == 0) {
        float q_total = 0.0f;
        float k_total = 0.0f;
        for (int warp = 0; warp < (blockDim.x + 31) / 32; ++warp) {
            q_total += reductions[warp];
            k_total += reductions[4 + warp];
        }
        reductions[0] = rsqrtf(q_total + eps);
        reductions[1] = rsqrtf(k_total + eps);
    }
    __syncthreads();
    if (lane < key_head_dim) {
        q_values[lane] *= reductions[0];
        k_values[lane] *= reductions[1];
        const int decay_index = vector_decay ? head * key_head_dim + lane : head;
        decay[lane] = safe_decay
            ? expf(sigmoid(expf(bf16_float(a_log[head])) *
                (bf16_float(a_row[decay_index]) + bf16_float(dt_bias[decay_index]))) *
                decay_lower_bound)
            : expf(-expf(bf16_float(a_log[head])) *
                softplus(bf16_float(a_row[decay_index]) + bf16_float(dt_bias[decay_index])));
    }
    __syncthreads();

    __nv_bfloat16* state = recurrent_state + static_cast<size_t>(head) *
        key_head_dim * value_head_dim;
    if (lane < value_head_dim) {
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + lane;
            state[offset] = __float2bfloat16(decay[k_dim] * bf16_float(state[offset]));
        }
        float memory = 0.0f;
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            memory += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + lane]) *
                k_values[k_dim];
        }
        const float delta = (v_value - memory) * sigmoid(bf16_float(b_row[head]));
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            const size_t offset = static_cast<size_t>(k_dim) * value_head_dim + lane;
            state[offset] = __float2bfloat16(bf16_float(state[offset]) +
                k_values[k_dim] * delta);
        }
        float value = 0.0f;
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            value += bf16_float(state[static_cast<size_t>(k_dim) * value_head_dim + lane]) *
                q_values[k_dim];
        }
        output_row[head * value_head_dim + lane] = __float2bfloat16(
            value / sqrtf(static_cast<float>(key_head_dim)));
    }
    __syncthreads();

    float output_square = 0.0f;
    for (int d = lane; d < value_head_dim; d += blockDim.x) {
        const float value = bf16_float(output_row[head * value_head_dim + d]);
        output_square += value * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        output_square += __shfl_down_sync(0xffffffffu, output_square, offset);
    }
    if ((lane & 31) == 0) reductions[lane >> 5] = output_square;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (int warp = 0; warp < (blockDim.x + 31) / 32; ++warp) total += reductions[warp];
        *inverse = rsqrtf(total / value_head_dim + eps);
    }
    __syncthreads();
    if (lane < value_head_dim) {
        const int offset = head * value_head_dim + lane;
        const float gate = bf16_float(z_row[offset]);
        output_row[offset] = __float2bfloat16(bf16_float(output_row[offset]) * *inverse *
            bf16_float(norm_weight[lane]) *
            (sigmoid_output_gate ? sigmoid(gate) : gate * sigmoid(gate)));
    }
    }
}

} // namespace

void launch_interleave_gated_delta_qkv(const __nv_bfloat16* q,
                                       const __nv_bfloat16* k,
                                       const __nv_bfloat16* v,
                                       __nv_bfloat16* qkv, int rows,
                                       int key_width, int value_width,
                                       cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) *
        static_cast<size_t>(2 * key_width + value_width);
    const int blocks = static_cast<int>(std::min<size_t>(
        4096, (total + 255) / 256));
    interleave_gated_delta_qkv_kernel<<<blocks, 256, 0, stream>>>(
        q, k, v, qkv, rows, key_width, value_width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gated_delta_net(const __nv_bfloat16* projected_qkv,
                            const __nv_bfloat16* projected_z,
                            const __nv_bfloat16* projected_b,
                            const __nv_bfloat16* projected_a,
                            const __nv_bfloat16* conv_weight,
                            const __nv_bfloat16* dt_bias,
                            const __nv_bfloat16* a_log,
                            const __nv_bfloat16* norm_weight,
                            __nv_bfloat16* conv_state,
                            __nv_bfloat16* recurrent_state,
                            __nv_bfloat16* output, int rows, int conv_kernel,
                            int key_head_dim, int value_head_dim, int key_heads,
                            int value_heads, float eps, bool vector_decay,
                            bool safe_decay, float decay_lower_bound,
                            bool sigmoid_output_gate, cudaStream_t stream) {
    if (rows > 0 && key_head_dim <= 256 && value_head_dim <= 256) {
        const int key_width = key_heads * key_head_dim;
        const int qkv_width = 2 * key_width + value_heads * value_head_dim;
        if (rows >= 64) {
            const int tiles = (value_head_dim + 3) / 4;
            gated_delta_sequence_prepare_kernel<<<key_heads, 256, 0, stream>>>(
                const_cast<__nv_bfloat16*>(projected_qkv), conv_weight, conv_state,
                rows, conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, eps);
            const dim3 state_grid(value_heads, tiles);
            const dim3 state_block(32, 4);
            switch (key_head_dim) {
                case 32:
                    gated_delta_sequence_state_kernel<32><<<state_grid, state_block, 0, stream>>>(
                        projected_qkv, projected_b, projected_a, dt_bias, a_log, recurrent_state,
                        output, rows, key_head_dim, value_head_dim, key_heads, value_heads,
                        vector_decay, safe_decay, decay_lower_bound);
                    break;
                case 64:
                    gated_delta_sequence_state_kernel<64><<<state_grid, state_block, 0, stream>>>(
                        projected_qkv, projected_b, projected_a, dt_bias, a_log, recurrent_state,
                        output, rows, key_head_dim, value_head_dim, key_heads, value_heads,
                        vector_decay, safe_decay, decay_lower_bound);
                    break;
                case 128:
                    gated_delta_sequence_state_kernel<128><<<state_grid, state_block, 0, stream>>>(
                        projected_qkv, projected_b, projected_a, dt_bias, a_log, recurrent_state,
                        output, rows, key_head_dim, value_head_dim, key_heads, value_heads,
                        vector_decay, safe_decay, decay_lower_bound);
                    break;
                case 256:
                    gated_delta_sequence_state_kernel<256><<<state_grid, state_block, 0, stream>>>(
                        projected_qkv, projected_b, projected_a, dt_bias, a_log, recurrent_state,
                        output, rows, key_head_dim, value_head_dim, key_heads, value_heads,
                        vector_decay, safe_decay, decay_lower_bound);
                    break;
                default:
                    gated_delta_sequence_state_kernel<<<state_grid, state_block, 0, stream>>>(
                        projected_qkv, projected_b, projected_a, dt_bias, a_log, recurrent_state,
                        output, rows, key_head_dim, value_head_dim, key_heads, value_heads,
                        vector_decay, safe_decay, decay_lower_bound);
                    break;
            }
            gated_delta_sequence_norm_kernel<<<value_heads, 128, 0, stream>>>(
                output, projected_z, norm_weight, rows, value_head_dim, value_heads,
                eps, sigmoid_output_gate);
            CELEG_KERNEL_DEBUG_SYNC(stream);
            return;
        }
        if (key_heads == value_heads) {
            const size_t shared_bytes = static_cast<size_t>(3 * key_head_dim + 9) * sizeof(float);
            gated_delta_fused_single_head_kernel<<<key_heads, 256, shared_bytes, stream>>>(
                projected_qkv, projected_z, projected_b, projected_a, conv_weight,
                dt_bias, a_log, norm_weight, conv_state, recurrent_state, output, rows,
                conv_kernel, key_head_dim, value_head_dim, key_heads, eps,
                vector_decay, safe_decay, decay_lower_bound, sigmoid_output_gate);
            CELEG_KERNEL_DEBUG_SYNC(stream);
            return;
        }
        gated_delta_conv_kernel<<<(qkv_width + 255) / 256, 256, 0, stream>>>(
            const_cast<__nv_bfloat16*>(projected_qkv), conv_weight, conv_state,
            qkv_width, conv_kernel);
        gated_delta_qk_norm_kernel<<<key_heads, 1, 0, stream>>>(
            const_cast<__nv_bfloat16*>(projected_qkv), key_head_dim, key_heads,
            key_width, eps);
        const size_t shared_bytes = static_cast<size_t>(key_head_dim + 1) * sizeof(float);
        gated_delta_recurrent_kernel<<<value_heads, 256, shared_bytes, stream>>>(
            projected_qkv, projected_z, projected_b, projected_a, dt_bias, a_log,
            norm_weight, recurrent_state, output, key_head_dim, value_head_dim,
            key_heads, value_heads, eps, vector_decay, safe_decay,
            decay_lower_bound, sigmoid_output_gate);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        return;
    }
    gated_delta_net_kernel<<<1, 1, 0, stream>>>(
        projected_qkv, projected_z, projected_b, projected_a, conv_weight,
        dt_bias, a_log, norm_weight, conv_state, recurrent_state, output,
        rows, conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, eps,
        vector_decay, safe_decay, decay_lower_bound, sigmoid_output_gate);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
