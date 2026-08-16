#include "celeg/backend/cpu/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace celeg {
namespace {

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

float softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return std::exp(value);
    return std::log1p(std::exp(value));
}

float silu(float value) {
    return value * sigmoid(value);
}

void validate_gated_delta_arguments(const float* projected_qkv, const float* projected_z,
                                    const float* projected_b, const float* projected_a,
                                    const float* conv_weight, const float* dt_bias,
                                    const float* a_log, const float* norm_weight,
                                    const float* conv_state, const float* recurrent_state,
                                    const float* output, int conv_kernel, int key_head_dim,
                                    int value_head_dim, int key_heads, int value_heads,
                                    float eps) {
    if (!projected_qkv || !projected_z || !projected_b || !projected_a ||
        !conv_weight || !dt_bias || !a_log || !norm_weight || !conv_state ||
        !recurrent_state || !output || conv_kernel <= 0 || key_head_dim <= 0 ||
        value_head_dim <= 0 || key_heads <= 0 || value_heads <= 0 ||
        value_heads % key_heads != 0 || eps <= 0.0f) {
        throw std::invalid_argument("invalid GatedDeltaNet arguments");
    }
}

struct GatedDeltaScratch {
    std::vector<float> filtered_qkv;
    std::vector<float> normalized_q;
    std::vector<float> normalized_k;
    std::vector<float> delta;
    std::vector<float> normalized_head;
    std::vector<float> decay_base;

    GatedDeltaScratch(int conv_dim, int key_width, int value_head_dim,
                      int value_heads, const float* a_log, bool a_log_needs_exp) :
        filtered_qkv(static_cast<size_t>(conv_dim)),
        normalized_q(static_cast<size_t>(key_width)),
        normalized_k(static_cast<size_t>(key_width)),
        delta(static_cast<size_t>(value_head_dim)),
        normalized_head(static_cast<size_t>(value_head_dim)),
        decay_base(static_cast<size_t>(value_heads)) {
        for (int head = 0; head < value_heads; ++head) {
            decay_base[static_cast<size_t>(head)] =
                a_log_needs_exp ? std::exp(a_log[head]) : a_log[head];
        }
    }
};

void gated_delta_step(const float* projected_qkv, const float* projected_z,
                      const float* projected_b, const float* projected_a,
                      const float* conv_weight, const float* dt_bias,
                      const float* a_log, const float* norm_weight,
                      float* conv_state, float* recurrent_state, float* output,
                      int conv_kernel, int key_head_dim, int value_head_dim,
                      int key_heads, int value_heads, float eps, bool vector_decay,
                      bool safe_decay, float decay_lower_bound,
                      bool sigmoid_output_gate, bool a_log_needs_exp,
                      GatedDeltaScratch& scratch) {
    const int key_width = key_heads * key_head_dim;
    const int value_width = value_heads * value_head_dim;
    const int conv_dim = 2 * key_width + value_width;
    std::vector<float>& filtered_qkv = scratch.filtered_qkv;

    for (int channel = 0; channel < conv_dim; ++channel) {
        float* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
        float filtered = 0.0f;
        const float* weight = conv_weight + static_cast<size_t>(channel) * conv_kernel;
        if (conv_kernel == 4) {
            history[0] = history[1];
            history[1] = history[2];
            history[2] = history[3];
            history[3] = projected_qkv[channel];
            filtered = history[0] * weight[0] + history[1] * weight[1] +
                       history[2] * weight[2] + history[3] * weight[3];
        } else if (conv_kernel == 3) {
            history[0] = history[1];
            history[1] = history[2];
            history[2] = projected_qkv[channel];
            filtered = history[0] * weight[0] + history[1] * weight[1] +
                       history[2] * weight[2];
        } else if (conv_kernel == 2) {
            history[0] = history[1];
            history[1] = projected_qkv[channel];
            filtered = history[0] * weight[0] + history[1] * weight[1];
        } else {
            for (int tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
            history[conv_kernel - 1] = projected_qkv[channel];
            for (int tap = 0; tap < conv_kernel; ++tap) {
                filtered += history[tap] * weight[tap];
            }
        }
        filtered_qkv[static_cast<size_t>(channel)] = silu(filtered);
    }

    const float* q = filtered_qkv.data();
    const float* k = filtered_qkv.data() + key_width;
    const float* v = filtered_qkv.data() + 2 * key_width;
    const int repeat = value_heads / key_heads;
    std::vector<float>& normalized_q = scratch.normalized_q;
    std::vector<float>& normalized_k = scratch.normalized_k;
    for (int head = 0; head < key_heads; ++head) {
        float q_norm = 0.0f;
        float k_norm = 0.0f;
        for (int d = 0; d < key_head_dim; ++d) {
            q_norm += q[head * key_head_dim + d] * q[head * key_head_dim + d];
            k_norm += k[head * key_head_dim + d] * k[head * key_head_dim + d];
        }
        q_norm = std::sqrt(q_norm + eps);
        k_norm = std::sqrt(k_norm + eps);
        for (int d = 0; d < key_head_dim; ++d) {
            normalized_q[head * key_head_dim + d] =
                q[head * key_head_dim + d] / q_norm;
            normalized_k[head * key_head_dim + d] =
                k[head * key_head_dim + d] / k_norm;
        }
    }

    const float inverse_key_norm = 1.0f / std::sqrt(static_cast<float>(key_head_dim));
    for (int value_head = 0; value_head < value_heads; ++value_head) {
        const int key_head = value_head / repeat;
        const float beta = sigmoid(projected_b[value_head]);
        float* __restrict state = recurrent_state + static_cast<size_t>(value_head) *
            key_head_dim * value_head_dim;
        float* __restrict delta = scratch.delta.data();
        const float* __restrict normalized_key = normalized_k.data() + key_head * key_head_dim;
        const float* __restrict normalized_query = normalized_q.data() + key_head * key_head_dim;
        const float* __restrict head_value = v + value_head * value_head_dim;
        const float decay_base = scratch.decay_base[static_cast<size_t>(value_head)];
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                const int decay_index = vector_decay ? key_head * key_head_dim + k_dim
                                                      : value_head;
                // When a_log is already the final log-domain coefficient (GGUF
                // convention: -exp(A_log) baked in at conversion time), it must be
                // used as-is; when it is the raw A_log parameter (safetensors
                // convention), the consumer must apply -exp() itself.
                const float decay = safe_decay
                    ? std::exp(sigmoid(decay_base *
                        (projected_a[decay_index] + dt_bias[decay_index])) *
                        decay_lower_bound)
                    : std::exp((a_log_needs_exp ? -decay_base : decay_base) *
                        softplus(projected_a[decay_index] + dt_bias[decay_index]));
                if (value_head == 0 && k_dim == 0 && std::getenv("CELEG_DEBUG_DECAY")) {
                    fprintf(stderr, "[gdn decay] needs_exp=%d decay_base=%.6f a=%.6f dt_bias=%.6f decay=%.6f\n",
                        (int)a_log_needs_exp, decay_base, projected_a[decay_index],
                        dt_bias[decay_index], decay);
                }
#pragma loop(ivdep)
                for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
                    state[k_dim * value_head_dim + v_dim] *= decay;
                }
        }
        for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
            float memory = 0.0f;
            for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                memory += state[k_dim * value_head_dim + v_dim] *
                    normalized_key[k_dim];
            }
            delta[v_dim] = (head_value[v_dim] - memory) * beta;
        }
        for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
            const float key_value = normalized_key[k_dim];
#pragma loop(ivdep)
            for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
                state[k_dim * value_head_dim + v_dim] += key_value * delta[v_dim];
            }
        }
        for (int v_dim = 0; v_dim < value_head_dim; ++v_dim) {
            float value = 0.0f;
            for (int k_dim = 0; k_dim < key_head_dim; ++k_dim) {
                value += state[k_dim * value_head_dim + v_dim] *
                    normalized_query[k_dim];
            }
            output[value_head * value_head_dim + v_dim] = value * inverse_key_norm;
        }
    }

    std::vector<float>& normalized_head = scratch.normalized_head;
    for (int value_head = 0; value_head < value_heads; ++value_head) {
        float* head_output = output + value_head * value_head_dim;
        cpu_rmsnorm(head_output, norm_weight, normalized_head.data(), value_head_dim, eps);
        const float* gate = projected_z + value_head * value_head_dim;
        for (int d = 0; d < value_head_dim; ++d) {
            head_output[d] = normalized_head[static_cast<size_t>(d)] *
                (sigmoid_output_gate ? sigmoid(gate[d]) : silu(gate[d]));
        }
    }
}

}

void cpu_gated_delta_net_decode(const float* projected_qkv, const float* projected_z,
                                const float* projected_b, const float* projected_a,
                                const float* conv_weight, const float* dt_bias,
                                const float* a_log, const float* norm_weight,
                                float* conv_state, float* recurrent_state,
                                float* output, int conv_kernel, int key_head_dim,
                                int value_head_dim, int key_heads, int value_heads,
                                float eps, bool vector_decay, bool safe_decay,
                                float decay_lower_bound, bool sigmoid_output_gate,
                                bool a_log_needs_exp) {
    validate_gated_delta_arguments(projected_qkv, projected_z, projected_b, projected_a,
        conv_weight, dt_bias, a_log, norm_weight, conv_state, recurrent_state, output,
        conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, eps);
    const int key_width = key_heads * key_head_dim;
    const int value_width = value_heads * value_head_dim;
    GatedDeltaScratch scratch(2 * key_width + value_width, key_width,
                              value_head_dim, value_heads, a_log, a_log_needs_exp);
    gated_delta_step(projected_qkv, projected_z, projected_b, projected_a, conv_weight,
        dt_bias, a_log, norm_weight, conv_state, recurrent_state, output, conv_kernel,
        key_head_dim, value_head_dim, key_heads, value_heads, eps, vector_decay,
        safe_decay, decay_lower_bound, sigmoid_output_gate, a_log_needs_exp, scratch);
}

void cpu_gated_delta_net_prefill(const float* projected_qkv, const float* projected_z,
                                 const float* projected_b, const float* projected_a,
                                 const float* conv_weight, const float* dt_bias,
                                 const float* a_log, const float* norm_weight,
                                 float* conv_state, float* recurrent_state,
                                 float* output, size_t rows, int conv_kernel,
                                 int key_head_dim, int value_head_dim, int key_heads,
                                 int value_heads, float eps, bool vector_decay,
                                 bool safe_decay, float decay_lower_bound,
                                 bool sigmoid_output_gate, bool a_log_needs_exp) {
    if (rows == 0) throw std::invalid_argument("GatedDeltaNet prefill needs rows");
    validate_gated_delta_arguments(projected_qkv, projected_z, projected_b, projected_a,
        conv_weight, dt_bias, a_log, norm_weight, conv_state, recurrent_state, output,
        conv_kernel, key_head_dim, value_head_dim, key_heads, value_heads, eps);
    const int qkv_width = 2 * key_heads * key_head_dim + value_heads * value_head_dim;
    const int z_width = value_heads * value_head_dim;
    GatedDeltaScratch scratch(qkv_width, key_heads * key_head_dim,
                              value_head_dim, value_heads, a_log, a_log_needs_exp);
    for (size_t row = 0; row < rows; ++row) {
        gated_delta_step(projected_qkv + row * static_cast<size_t>(qkv_width),
            projected_z + row * static_cast<size_t>(z_width),
            projected_b + row * static_cast<size_t>(value_heads),
            projected_a + row * static_cast<size_t>(vector_decay
                ? key_heads * key_head_dim : value_heads), conv_weight, dt_bias,
            a_log, norm_weight, conv_state, recurrent_state,
            output + row * static_cast<size_t>(z_width), conv_kernel, key_head_dim,
            value_head_dim, key_heads, value_heads, eps, vector_decay, safe_decay,
            decay_lower_bound, sigmoid_output_gate, a_log_needs_exp, scratch);
    }
}

}
