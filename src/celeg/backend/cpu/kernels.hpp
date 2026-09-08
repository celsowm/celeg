#pragma once

#include "celeg/backend/cpu/linear.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/model/definition.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace celeg {

void cpu_rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps);
void cpu_rmsnorm_inplace(float* data, const float* weight,
                         size_t width, float eps);
void cpu_residual_add(float* data, const float* residual, size_t count);
void cpu_swiglu(const float* gate_up, float* output, size_t count);
void cpu_gated_gelu_tanh(const float* gate_up, float* output, size_t count);
void cpu_relu2(const float* input, float* output, size_t count);
void cpu_gelu_tanh(float* data, size_t count);
void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      const RopePositionSpec& rope, float eps);
void cpu_qk_norm_only(float* data, const float* norm_weight,
                      int heads, int head_dim, float eps);
void cpu_rope(float* data, int heads, int head_dim, int position,
              const RopePositionSpec& rope);
void cpu_qk_norm_rope_mrope(float* data, const float* norm_weight,
                            int heads, int head_dim,
                            const std::array<int32_t, 3>& positions,
                            const std::array<int, 3>& sections,
                            bool interleaved, const RopePositionSpec& rope,
                            float eps);
void cpu_rope_mrope(float* data, int heads, int head_dim,
                    const std::array<int32_t, 3>& positions,
                    const std::array<int, 3>& sections,
                    bool interleaved, const RopePositionSpec& rope);
void cpu_gqa_decode(const float* q, const float* key_cache,
                    const float* value_cache, float* output,
                    int sequence_length, int q_heads, int kv_heads,
                    int head_dim);
void cpu_gqa_decode_bf16(const float* q, const uint16_t* key_cache,
                         const uint16_t* value_cache, float* output,
                         int sequence_length, int q_heads, int kv_heads,
                         int head_dim);
void cpu_conv_decode(const float* projected_bcx, const float* weight,
                     float* state, float* output, int hidden,
                     int cache_length, int position);
void cpu_conv_prefill(const float* projected_bcx, const float* weight,
                      float* state, float* output, size_t rows, int hidden,
                      int cache_length, int base_position,
                      CpuThreadPool& thread_pool);
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
