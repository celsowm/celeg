#pragma once

#include "celeg/quantization/ggml.hpp"

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

    bool supports_width(int cols) const noexcept {
        if (cols <= 0) return false;
        const GgmlTypeTrait trait = ggml_type_trait(source);
        return trait.block_size != 0 &&
               cols % static_cast<int>(trait.block_size) == 0;
    }
};

std::optional<MetalLinearBinding> metal_linear_binding(GgmlType type);

}
