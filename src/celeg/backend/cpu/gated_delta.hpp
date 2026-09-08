#pragma once

#include <cstddef>

namespace celeg {

void cpu_gated_delta_net_decode(const float* projected_qkv, const float* projected_z,
                                const float* projected_b, const float* projected_a,
                                const float* conv_weight, const float* dt_bias,
                                const float* a_log, const float* norm_weight,
                                float* conv_state, float* recurrent_state,
                                float* output, int conv_kernel, int key_head_dim,
                                int value_head_dim, int key_heads, int value_heads,
                                float eps, bool vector_decay = false,
                                bool safe_decay = false, float decay_lower_bound = -5.0f,
                                bool sigmoid_output_gate = false,
                                bool a_log_needs_exp = true);
void cpu_gated_delta_net_prefill(const float* projected_qkv, const float* projected_z,
                                 const float* projected_b, const float* projected_a,
                                 const float* conv_weight, const float* dt_bias,
                                 const float* a_log, const float* norm_weight,
                                 float* conv_state, float* recurrent_state,
                                 float* output, size_t rows, int conv_kernel,
                                 int key_head_dim, int value_head_dim, int key_heads,
                                 int value_heads, float eps, bool vector_decay = false,
                                 bool safe_decay = false, float decay_lower_bound = -5.0f,
                                 bool sigmoid_output_gate = false,
                                 bool a_log_needs_exp = true);

}
