#pragma once

#include "celeg/quantization/ggml.hpp"

#include <cstdint>
#include <optional>

namespace celeg::metal_model_detail {

enum class MetalLinearStorage {
    Float32,
    Float16,
    BFloat16,
    Q4_0,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
};

struct MetalLinearBinding {
    GgmlType source = GgmlType::Unknown;
    MetalLinearStorage storage = MetalLinearStorage::Float32;
    std::uint32_t block_size = 0;

    bool supports_width(int cols) const noexcept {
        return cols > 0 && block_size != 0 &&
               cols % static_cast<int>(block_size) == 0;
    }
};

std::optional<MetalLinearBinding> metal_linear_binding(GgmlType type);

}
