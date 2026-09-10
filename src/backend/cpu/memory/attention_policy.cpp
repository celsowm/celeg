#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/attention/bias_semantics.hpp"
#include "celeg/attention/pattern_semantics.hpp"

#include <algorithm>
#include <type_traits>

namespace celeg {

bool CpuAttentionPattern::allows(int query_position, int key_position) const {
    return std::visit([&](const auto& value) -> bool {
        using Pattern = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Pattern, FullCausalPattern>) {
            return attention_semantics::causal_visible(query_position, key_position);
        } else if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
            return attention_semantics::sliding_window_visible(
                query_position, key_position, value.window);
        } else if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
            return attention_semantics::bidirectional_visible(
                query_position, key_position);
        } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
            return attention_semantics::prefix_lm_visible(
                query_position, key_position, value.prefix_length);
        } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
            return attention_semantics::block_sparse_visible(
                query_position, key_position, value.block_size,
                value.local_blocks, value.global_blocks);
        } else if constexpr (std::is_same_v<Pattern, DynamicSparsePattern>) {
            throw std::invalid_argument(
                "CPU DynamicSparse requires content-ranked selected-block context");
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
            return attention_semantics::prefix_lm_may_read_future(
                query_position, sequence_length, value.prefix_length);
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
            return attention_semantics::sliding_window_first_candidate(
                query_position, value.window);
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
    const int directional_bucket_count = relative.bidirectional
        ? relative.bucket_count / 2 : relative.bucket_count;
    if (directional_bucket_count <= 0) {
        throw std::invalid_argument("relative position bias bucket count is invalid");
    }
    const int bucket = attention_semantics::relative_position_bucket(
        query_position, key_position, relative.bucket_count,
        relative.max_distance, relative.bidirectional);
    return relative.values[static_cast<size_t>(query_head) *
                           static_cast<size_t>(relative.bucket_count) +
                           static_cast<size_t>(bucket)];
}

float score_alibi_bias(const CpuAlibiBiasView& alibi, int query_head,
                       int query_position, int key_position) {
    if (query_head < 0 || static_cast<size_t>(query_head) >= alibi.slope_count) {
        throw std::invalid_argument("ALiBi query head is out of range");
    }
    return attention_semantics::alibi_bias(
        alibi.slopes[static_cast<size_t>(query_head)],
        query_position, key_position);
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
