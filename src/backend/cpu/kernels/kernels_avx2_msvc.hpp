#pragma once

// MSVC has no equivalent of GCC/Clang's per-function
// __attribute__((target("..."))), so the AVX2 kernel bodies for MSVC live in
// their own translation unit (kernels_avx2_msvc.cpp) instead of being
// multiversioned inline alongside the scalar fallback in kernels.cpp. These
// declarations are the seam kernels.cpp dispatches through on MSVC.

#include <cstddef>
#include <cstdint>

namespace lfm::detail {

float q4_dot_avx2_msvc(const uint8_t* packed_row,
                        const uint16_t* scales_bf16,
                        const float* activation,
                        size_t cols,
                        size_t group_size,
                        size_t groups_per_row);

void cpu_rmsnorm_avx2_msvc(const float* input, const float* weight, float* output,
                            size_t width, float eps);

void cpu_residual_add_avx2_msvc(float* data, const float* residual, size_t count);

void cpu_swiglu_avx2_msvc(const float* gate_up, float* output, size_t count);

void cpu_qk_norm_rope_avx2_msvc(float* data, const float* norm_weight,
                                 const float* cos_vals, const float* sin_vals,
                                 int heads, int head_dim, float eps);

} // namespace lfm::detail
