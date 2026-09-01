#pragma once

#include "celeg/model/graph.hpp"

#include <cstddef>
#include <stdexcept>

namespace celeg {

inline bool cuda_packed_attention_is_unsupported(
    const AttentionSpec& attention) {
    return std::holds_alternative<BidirectionalPattern>(attention.pattern) ||
           std::holds_alternative<BlockSparsePattern>(attention.pattern) ||
           std::holds_alternative<DynamicSparsePattern>(attention.pattern);
}

inline void validate_cuda_packed_attention(
    const AttentionSpec& attention) {
    if (cuda_packed_attention_is_unsupported(attention)) {
        throw std::invalid_argument(
            "CUDA packed attention does not implement this visibility pattern");
    }
}

inline void validate_cuda_packed_prefill_span(
    const AttentionSpec& attention,
    int start_position,
    std::size_t token_count) {
    const auto* prefix = std::get_if<PrefixLmPattern>(&attention.pattern);
    if (!prefix || start_position >= prefix->prefix_length) return;

    const std::size_t end_position =
        static_cast<std::size_t>(start_position) + token_count;
    if (start_position != 0 ||
        end_position < static_cast<std::size_t>(prefix->prefix_length)) {
        throw std::invalid_argument(
            "CUDA packed Prefix-LM prefill must materialize the complete prefix in its first span");
    }
}

inline void validate_cuda_packed_decode_position(
    const AttentionSpec& attention,
    int position) {
    const auto* prefix = std::get_if<PrefixLmPattern>(&attention.pattern);
    if (prefix && position < prefix->prefix_length) {
        throw std::invalid_argument(
            "CUDA packed Prefix-LM decode requires a completed prefix");
    }
}

inline int cuda_packed_prefix_length(const AttentionSpec& attention) {
    const auto* prefix = std::get_if<PrefixLmPattern>(&attention.pattern);
    return prefix ? prefix->prefix_length : 0;
}

}
