#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace celeg {

struct CudaPrefillSpanDecision {
    std::size_t count = 0;
    bool defer = false;
    bool requires_packed = false;
};

inline CudaPrefillSpanDecision plan_cuda_prefill_span(
    std::size_t remaining,
    std::size_t prefill_offset,
    std::size_t required_initial_span,
    int chunk_tokens,
    int token_budget,
    bool batch_has_work) {
    if (chunk_tokens <= 0) {
        throw std::invalid_argument("CUDA prefill chunk size must be positive");
    }
    if (remaining == 0) return {};

    const std::size_t ordinary = std::min<std::size_t>(
        remaining,
        static_cast<std::size_t>(std::max(
            0, std::min(chunk_tokens, token_budget))));

    if (prefill_offset != 0 || required_initial_span == 0) {
        return {.count = ordinary};
    }
    if (remaining < required_initial_span) {
        throw std::invalid_argument(
            "CUDA Prefix-LM prompt is shorter than the required prefix");
    }
    if (batch_has_work && token_budget < static_cast<int>(required_initial_span)) {
        return {.defer = true, .requires_packed = true};
    }
    return {
        .count = std::max(required_initial_span, ordinary),
        .defer = false,
        .requires_packed = true};
}

}