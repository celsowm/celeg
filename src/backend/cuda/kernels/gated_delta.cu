#include "kernel_common.cuh"
#include "celeg/backend/cuda/kernels/gated_delta.hpp"

#include <algorithm>
#include <cmath>

namespace celeg {
namespace {

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

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ float softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
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
    gated_delta_net_kernel<<<1, 1, 0, stream>>>(
        projected_qkv, projected_z, projected_b, projected_a, conv_weight,
        dt_bias, a_log, norm_weight, conv_state, recurrent_state, output,
        rows, conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, eps,
        vector_decay, safe_decay, decay_lower_bound, sigmoid_output_gate);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
