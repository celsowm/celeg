#pragma once

#include "lfm/backend/cpu/isa.hpp"
#include "lfm/backend/cpu/quantization.hpp"
#include "lfm/backend/cpu/thread_pool.hpp"

#include <cstddef>
#include <cstdint>

namespace lfm {

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

class CpuLinearEngine {
public:
    CpuLinearEngine(CpuIsa isa, CpuThreadPool& pool);

    CpuIsa isa() const { return isa_; }
    void gemv(const Q4GroupMatrix& weight, const float* input, float* output,
              float beta = 0.0f) const;
    void gemm(const Q4GroupMatrix& weight, const float* input, float* output,
              size_t rows, float beta = 0.0f) const;
    void embedding(const Q4GroupMatrix& table, int32_t token, float* output) const;

private:
    CpuIsa isa_;
    CpuThreadPool* pool_;
    Q4DotFunction dot_;
    Q4Q8DotFunction q8_dot_ = nullptr;
    bool dynamic_q8_ = false;
};

void cpu_rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps);
void cpu_rmsnorm_inplace(float* data, const float* weight,
                         size_t width, float eps);
void cpu_residual_add(float* data, const float* residual, size_t count);
void cpu_swiglu(const float* gate_up, float* output, size_t count);
void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      float rope_theta, float eps);
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

} // namespace lfm
