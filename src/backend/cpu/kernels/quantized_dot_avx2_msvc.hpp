#pragma once

#include "celeg/backend/cpu/quantization.hpp"

#include <cstddef>
#include <cstdint>

namespace celeg::detail {

float f32_dot_avx2_msvc(const float* weight, const float* activation,
                         size_t cols);

float q4_dot_avx2_msvc(const uint8_t* packed_row,
                       const uint16_t* scales_bf16,
                       const float* activation,
                       size_t cols, size_t group_size, size_t groups_per_row);

float q4_q8_dot_avx2_msvc(const uint8_t* packed_row,
                          const uint16_t* weight_scales_bf16,
                          const int8_t* activation_q8,
                          const float* activation_scales,
                          const int32_t* activation_sums,
                          size_t cols, size_t group_size, size_t groups_per_row);

void q4_q8_dot4_avx2_msvc(const uint8_t* packed_row,
                           const uint16_t* weight_scales_bf16,
                           const int8_t* activation_values, size_t value_stride,
                           const float* activation_scales,
                           const int32_t* activation_sums, size_t group_stride,
                           size_t cols, size_t group_size, size_t groups_per_row,
                           float* output4);

}
