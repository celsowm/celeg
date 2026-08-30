#pragma once

#include "celeg/checkpoint/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace celeg {

std::size_t tensor_element_count(std::span<const std::int64_t> shape,
                                 std::string_view context);

bool tensor_shape_matches(std::span<const std::int64_t> actual,
                          std::span<const std::int64_t> expected);

bool tensor_shape_is_compatible(std::span<const std::int64_t> actual,
                                std::span<const std::int64_t> expected);

std::vector<float> decode_tensor_f32(const HostTensorView& tensor,
                                     std::span<const std::int64_t> expected,
                                     std::string_view name);

}
