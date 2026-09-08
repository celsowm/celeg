#pragma once

#include "celeg/backend/cpu/isa.hpp"

#include <cstddef>
#include <cstdint>

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

}
