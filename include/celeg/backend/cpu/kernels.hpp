#pragma once

#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/backend/cpu/thread_pool.hpp"
#include "celeg/model/definition.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <vector>

namespace celeg {

using Q4DotFunction = float (*)(const uint8_t* packed_row,
                                const uint16_t* scales_bf16,
                                const float* activation,
                                size_t cols,
                                size_t group_size,
                                size_t groups_per_row);

Q4DotFunction select_q4_dot_kernel(CpuIsa isa);

using Q4Q8DotFunction = float (*)(const uint8_t* packed_row,
                                  const uint16_t* weight_scales_bf16,
                                  const int8_t* activation_q8,
                                  const float* activation_scales,
                                  const int32_t* activation_sums,
                                  size_t cols,
                                  size_t group_size,
                                  size_t groups_per_row);

Q4Q8DotFunction select_q4_q8_dot_kernel(CpuIsa isa);
float q4_q8_dot_scalar(const uint8_t* packed_row,
                       const uint16_t* weight_scales_bf16,
                       const int8_t* activation_q8,
                       const float* activation_scales,
                       const int32_t* activation_sums,
                       size_t cols,
                       size_t group_size,
                       size_t groups_per_row);
float q4_dot_scalar(const uint8_t* packed_row,
                    const uint16_t* scales_bf16,
                    const float* activation,
                    size_t cols,
                    size_t group_size,
                    size_t groups_per_row);

// Internal routed-MoE GEMM descriptor. Rows are packed contiguously across
// jobs; all jobs in one invocation must have identical matrix dimensions.
struct CpuGroupedGemmJob {
    const CpuLinearWeight* weight = nullptr;
    size_t row_offset = 0;
    size_t rows = 0;
};

class CpuLinearEngine {
public:
    CpuLinearEngine(CpuIsa isa, CpuThreadPool& pool);

    CpuIsa isa() const { return isa_; }
    void gemv(const Q4GroupMatrix& weight, const float* input, float* output,
              float beta = 0.0f) const;
    void gemm(const Q4GroupMatrix& weight, const float* input, float* output,
              size_t rows, float beta = 0.0f) const;
    void embedding(const Q4GroupMatrix& table, int32_t token, float* output) const;
    void gemv(const CpuLinearWeight& weight, const float* input, float* output,
              float beta = 0.0f) const;
    void gemm(const CpuLinearWeight& weight, const float* input, float* output,
              size_t rows, float beta = 0.0f) const;
    void prepare_gguf_activation(const float* input, size_t rows, size_t cols,
                                 std::vector<CpuQ8KBlock>& activation) const;
    void gemm_gguf(std::span<const CpuQ8KBlock> activation,
                   const CpuLinearWeight& weight, float* output,
                   size_t rows, float beta = 0.0f) const;
    void gemm_grouped(std::span<const CpuGroupedGemmJob> jobs,
                      const float* input, float* output) const;
    void embedding(const CpuLinearWeight& table, int32_t token,
                   float* output) const;
    // Raw GEMV for pre-packed weight buffers (e.g. MoE router): output[n] = weight[n,k] @ input[k].
    void gemv_raw(const float* weight, const float* input, float* output,
                  int n, int k) const;
    // Batched raw router projection. Input is [rows,k], output is [rows,n].
    void gemm_raw(const float* weight, const float* input, float* output,
                  size_t rows, int n, int k) const;

private:
    CpuIsa isa_;
    CpuThreadPool* pool_;
    Q4DotFunction dot_;
    Q4Q8DotFunction q8_dot_ = nullptr;
    CpuGgufDotFunction gguf_dot_ = nullptr;
    CpuGgufDot4Function gguf_dot4_ = nullptr;
    bool dynamic_q8_ = false;
};

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
                                bool sigmoid_output_gate = false);
void cpu_gated_delta_net_prefill(const float* projected_qkv, const float* projected_z,
                                 const float* projected_b, const float* projected_a,
                                 const float* conv_weight, const float* dt_bias,
                                 const float* a_log, const float* norm_weight,
                                 float* conv_state, float* recurrent_state,
                                 float* output, size_t rows, int conv_kernel,
                                 int key_head_dim, int value_head_dim, int key_heads,
                                 int value_heads, float eps, bool vector_decay = false,
                                 bool safe_decay = false, float decay_lower_bound = -5.0f,
                                 bool sigmoid_output_gate = false);

} // namespace celeg
