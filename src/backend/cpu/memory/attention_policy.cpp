#include "celeg/backend/cpu/paged_kv.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace celeg {

bool CpuAttentionPattern::allows(int query_position, int key_position) const {
    if (query_position < 0 || key_position < 0) return false;
    return std::visit([&](const auto& value) -> bool {
        using Pattern = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Pattern, FullCausalPattern>) {
            return key_position <= query_position;
        } else if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
            return key_position <= query_position &&
                   key_position >= query_position - value.window + 1;
        } else if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
            return true;
        } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
            return query_position < value.prefix_length
                ? key_position < value.prefix_length
                : key_position <= query_position;
        } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
            const int query_block = query_position / value.block_size;
            const int key_block = key_position / value.block_size;
            if (key_block < value.global_blocks) return key_block <= query_block;
            return key_block <= query_block &&
                   key_block >= query_block - value.local_blocks + 1;
        } else if constexpr (std::is_same_v<Pattern, DynamicSparsePattern>) {
            const int query_block = query_position / value.block_size;
            const int key_block = key_position / value.block_size;
            if (key_block > query_block) return false;
            if (key_block == query_block) return true;
            return key_block < value.max_selected_blocks;
        } else {
            static_assert(always_false_v<Pattern>, "unhandled attention pattern variant");
        }
    }, storage);
}

bool CpuAttentionPattern::may_read_future(int query_position,
                                          int sequence_length) const {
    return std::visit([&](const auto& value) -> bool {
        using Pattern = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
            return true;
        } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
            return query_position < value.prefix_length &&
                   value.prefix_length < sequence_length;
        } else if constexpr (std::is_same_v<Pattern, FullCausalPattern> ||
                             std::is_same_v<Pattern, SlidingWindowPattern> ||
                             std::is_same_v<Pattern, BlockSparsePattern> ||
                             std::is_same_v<Pattern, DynamicSparsePattern>) {
            return false;
        } else {
            static_assert(always_false_v<Pattern>, "unhandled attention pattern variant");
        }
    }, storage);
}

int CpuAttentionPattern::first_candidate(int query_position) const {
    return std::visit([&](const auto& value) -> int {
        using Pattern = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
            return std::max(0, query_position - value.window + 1);
        } else if constexpr (std::is_same_v<Pattern, FullCausalPattern> ||
                             std::is_same_v<Pattern, BidirectionalPattern> ||
                             std::is_same_v<Pattern, PrefixLmPattern> ||
                             std::is_same_v<Pattern, BlockSparsePattern> ||
                             std::is_same_v<Pattern, DynamicSparsePattern>) {
            return 0;
        } else {
            static_assert(always_false_v<Pattern>, "unhandled attention pattern variant");
        }
    }, storage);
}

CpuAttentionBias CpuAttentionBias::lower(const AttentionBiasSpec& bias,
                                         std::span<const float> relative_values,
                                         int query_heads) {
    if (const auto* alibi = std::get_if<AlibiBiasSpec>(&bias)) {
        return {CpuAlibiBiasView{alibi->slopes.data(), alibi->slopes.size()}};
    }
    if (const auto* relative = std::get_if<RelativePositionBiasSpec>(&bias)) {
        if (query_heads <= 0 || relative->bucket_count <= 0 ||
            relative->max_distance <= 0 || relative_values.size() !=
                static_cast<size_t>(query_heads) *
                    static_cast<size_t>(relative->bucket_count)) {
            throw std::invalid_argument("relative position bias dimensions are invalid");
        }
        return {CpuRelativeBiasView{relative_values.data(), relative->bucket_count,
                                    relative->max_distance, relative->bidirectional}};
    }
    return {};
}

namespace {

float score_relative_bias(const CpuRelativeBiasView& relative, int query_head,
                          int query_position, int key_position) {
    if (query_head < 0) {
        throw std::invalid_argument("relative position bias query head is out of range");
    }
    const int relative_position = key_position - query_position;
    const int bucket_count = relative.bidirectional
        ? relative.bucket_count / 2 : relative.bucket_count;
    if (bucket_count <= 0) {
        throw std::invalid_argument("relative position bias bucket count is invalid");
    }
    const bool positive = relative.bidirectional && relative_position > 0;
    const int distance = relative.bidirectional
        ? std::abs(relative_position) : std::max(-relative_position, 0);
    const int max_exact = bucket_count / 2;
    int bucket = 0;
    if (distance < max_exact) {
        bucket = distance;
    } else {
        const float denominator = std::log(
            static_cast<float>(std::max(relative.max_distance, max_exact + 1)) /
            static_cast<float>(std::max(max_exact, 1)));
        const float logarithmic = denominator == 0.0f ? 0.0f : std::log(
            static_cast<float>(std::max(distance, max_exact)) /
            static_cast<float>(std::max(max_exact, 1))) / denominator;
        bucket = max_exact + static_cast<int>(
            logarithmic * static_cast<float>(bucket_count - max_exact));
        bucket = std::min(bucket, bucket_count - 1);
    }
    if (positive) bucket += bucket_count;
    return relative.values[static_cast<size_t>(query_head) *
                           static_cast<size_t>(relative.bucket_count) +
                           static_cast<size_t>(bucket)];
}

float score_alibi_bias(const CpuAlibiBiasView& alibi, int query_head,
                       int query_position, int key_position) {
    if (query_head < 0 || static_cast<size_t>(query_head) >= alibi.slope_count) {
        throw std::invalid_argument("ALiBi query head is out of range");
    }
    return -alibi.slopes[static_cast<size_t>(query_head)] *
        static_cast<float>(std::abs(query_position - key_position));
}

}

float CpuAttentionBias::score(int query_head, int query_position,
                              int key_position) const {
    return std::visit([&](const auto& value) -> float {
        using View = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<View, CpuNoAttentionBiasView>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<View, CpuAlibiBiasView>) {
            return score_alibi_bias(value, query_head, query_position, key_position);
        } else if constexpr (std::is_same_v<View, CpuRelativeBiasView>) {
            return score_relative_bias(value, query_head, query_position, key_position);
        } else {
            static_assert(always_false_v<View>, "unhandled CPU attention bias variant");
        }
    }, storage);
}

}
