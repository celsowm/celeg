#include "linear_dispatch.hpp"

#include <stdexcept>
#include <type_traits>

namespace celeg::detail {

LinearStorageKind classify_linear_weight(const CpuLinearWeight& weight) {
    bool any_q4 = false;
    bool any_int8 = false;
    bool any_gguf = false;
    bool any_bf16 = false;
    for (const CpuLinearMatrix& segment : weight.segments) {
        if (std::holds_alternative<Q4GroupMatrix>(segment)) any_q4 = true;
        else if (std::holds_alternative<CpuInt8Matrix>(segment)) any_int8 = true;
        else if (std::holds_alternative<CpuBf16Matrix>(segment)) any_bf16 = true;
        else any_gguf = true;
    }
    if (any_q4 && !any_int8 && !any_gguf && !any_bf16) return LinearStorageKind::Q4;
    if (any_int8 && !any_q4 && !any_gguf && !any_bf16) return LinearStorageKind::Int8;
    if (any_gguf && !any_q4 && !any_int8 && !any_bf16) return LinearStorageKind::Gguf;
    if (any_bf16 && !any_q4 && !any_int8 && !any_gguf) return LinearStorageKind::Bf16;
    throw std::logic_error("mixed CPU linear storage (Q4/INT8/GGUF/BF16) is unsupported");
}

size_t linear_segment_rows(const CpuLinearMatrix& segment) {
    return std::visit([](const auto& matrix) {
        return static_cast<size_t>(matrix.rows);
    }, segment);
}

}
