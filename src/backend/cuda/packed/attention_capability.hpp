#pragma once

#include "celeg/model/graph.hpp"

#include <stdexcept>

namespace celeg {

inline bool cuda_packed_prefill_requires_special_visibility(
    const AttentionSpec& attention) {
    return std::holds_alternative<BidirectionalPattern>(attention.pattern) ||
           std::holds_alternative<PrefixLmPattern>(attention.pattern) ||
           std::holds_alternative<BlockSparsePattern>(attention.pattern) ||
           std::holds_alternative<DynamicSparsePattern>(attention.pattern);
}

inline void validate_cuda_packed_prefill_attention(
    const AttentionSpec& attention) {
    if (cuda_packed_prefill_requires_special_visibility(attention)) {
        throw std::invalid_argument(
            "CUDA packed prefill does not implement constrained attention visibility");
    }
}

}
