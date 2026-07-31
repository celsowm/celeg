#pragma once

#include <cstddef>

namespace celeg::detail {

void cpu_rmsnorm_avx2_msvc(const float* input, const float* weight, float* output,
                           size_t width, float eps);
void cpu_residual_add_avx2_msvc(float* data, const float* residual, size_t count);
void cpu_swiglu_avx2_msvc(const float* gate_up, float* output, size_t count);
void cpu_qk_norm_rope_avx2_msvc(float* data, const float* norm_weight,
                                const float* cos_vals, const float* sin_vals,
                                int heads, int head_dim, float eps);

} // namespace celeg::detail
