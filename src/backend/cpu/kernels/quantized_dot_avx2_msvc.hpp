#pragma once

#include <cstddef>
#include <cstdint>

namespace lfm::detail {

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

} // namespace lfm::detail
