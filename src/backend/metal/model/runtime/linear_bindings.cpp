#include "linear_bindings.hpp"

namespace celeg::metal_model_detail {
namespace {

constexpr MetalLinearBinding kBindings[] = {
    {GgmlType::Q4_0, MetalLinearStorage::Q4_0, 32},
    {GgmlType::Q4_K, MetalLinearStorage::Q4K, 256},
    {GgmlType::Q5_K, MetalLinearStorage::Q5K, 256},
    {GgmlType::Q6_K, MetalLinearStorage::Q6K, 256},
    {GgmlType::Q8_0, MetalLinearStorage::Q8_0, 32},
};

}

std::optional<MetalLinearBinding> metal_linear_binding(GgmlType type) {
    for (const MetalLinearBinding& binding : kBindings) {
        if (binding.source == type) return binding;
    }
    return std::nullopt;
}

}
