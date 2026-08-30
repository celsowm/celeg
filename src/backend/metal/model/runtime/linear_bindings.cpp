#include "linear_bindings.hpp"

namespace celeg::metal_model_detail {
namespace {

constexpr MetalLinearBinding kBindings[] = {
    {GgmlType::Q4_0, MetalLinearStorage::Q4_0},
    {GgmlType::Q4_K, MetalLinearStorage::Q4K},
    {GgmlType::Q5_K, MetalLinearStorage::Q5K},
    {GgmlType::Q6_K, MetalLinearStorage::Q6K},
    {GgmlType::Q8_0, MetalLinearStorage::Q8_0},
};

}

std::optional<MetalLinearBinding> metal_linear_binding(GgmlType type) {
    for (const MetalLinearBinding& binding : kBindings) {
        if (binding.source == type) return binding;
    }
    return std::nullopt;
}

}
