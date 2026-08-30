#include "backend/metal/model/runtime/linear_bindings.hpp"

int main() {
    using celeg::GgmlType;
    using celeg::metal_model_detail::MetalLinearStorage;
    using celeg::metal_model_detail::metal_linear_binding;

    const auto q4_0 = metal_linear_binding(GgmlType::Q4_0);
    const auto q4_k = metal_linear_binding(GgmlType::Q4_K);
    const auto q5_k = metal_linear_binding(GgmlType::Q5_K);
    const auto q6_k = metal_linear_binding(GgmlType::Q6_K);
    const auto q8_0 = metal_linear_binding(GgmlType::Q8_0);
    if (!q4_0 || q4_0->storage != MetalLinearStorage::Q4_0 ||
        !q4_k || q4_k->storage != MetalLinearStorage::Q4K ||
        !q5_k || q5_k->storage != MetalLinearStorage::Q5K ||
        !q6_k || q6_k->storage != MetalLinearStorage::Q6K ||
        !q8_0 || q8_0->storage != MetalLinearStorage::Q8_0) return 1;
    if (!q4_0->supports_width(32) || q4_0->supports_width(31) ||
        !q4_k->supports_width(256) || q4_k->supports_width(128)) return 2;
    if (metal_linear_binding(GgmlType::Q4_1) ||
        metal_linear_binding(GgmlType::Unknown)) return 3;
    return 0;
}
