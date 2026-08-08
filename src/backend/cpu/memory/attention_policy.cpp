#include "celeg/backend/cpu/paged_kv.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace celeg {

CpuAttentionPattern CpuAttentionPattern::lower(const AttentionPatternSpec& pattern) {
    return std::visit([](const auto& value) -> CpuAttentionPattern {
        using Pattern = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Pattern, FullCausalPattern>) {
            return {CpuAttentionPatternKind::FullCausal};
        } else if constexpr (std::is_same_v<Pattern, SlidingWindowPattern>) {
            return {CpuAttentionPatternKind::SlidingWindow, value.window};
        } else if constexpr (std::is_same_v<Pattern, BidirectionalPattern>) {
            return {CpuAttentionPatternKind::Bidirectional};
        } else if constexpr (std::is_same_v<Pattern, PrefixLmPattern>) {
            return {CpuAttentionPatternKind::PrefixLm, 0, value.prefix_length};
        } else if constexpr (std::is_same_v<Pattern, BlockSparsePattern>) {
            return {CpuAttentionPatternKind::BlockSparse, 0, 0,
                    value.block_size, value.local_blocks, value.global_blocks};
        } else {
            return {CpuAttentionPatternKind::DynamicSparse, 0, 0,
                    value.block_size, 0, 0, value.max_selected_blocks};
        }
    }, pattern);
}

bool CpuAttentionPattern::allows(int query_position, int key_position) const {
    if (query_position < 0 || key_position < 0) return false;
    switch (kind) {
    case CpuAttentionPatternKind::FullCausal:
        return key_position <= query_position;
    case CpuAttentionPatternKind::SlidingWindow:
        return key_position <= query_position &&
               key_position >= query_position - window + 1;
    case CpuAttentionPatternKind::Bidirectional:
        return true;
    case CpuAttentionPatternKind::PrefixLm:
        return query_position < prefix_length
            ? key_position < prefix_length
            : key_position <= query_position;
    case CpuAttentionPatternKind::BlockSparse: {
        const int query_block = query_position / block_size;
        const int key_block = key_position / block_size;
        if (key_block < global_blocks) return key_block <= query_block;
        return key_block <= query_block &&
               key_block >= query_block - local_blocks + 1;
    }
    case CpuAttentionPatternKind::DynamicSparse: {
        const int query_block = query_position / block_size;
        const int key_block = key_position / block_size;
        if (key_block > query_block) return false;
        if (key_block == query_block) return true;
        return key_block < max_selected_blocks;
    }
    }
    return false;
}

bool CpuAttentionPattern::may_read_future(int query_position,
                                          int sequence_length) const {
    if (kind == CpuAttentionPatternKind::Bidirectional) return true;
    return kind == CpuAttentionPatternKind::PrefixLm &&
           query_position < prefix_length && prefix_length < sequence_length;
}

int CpuAttentionPattern::first_candidate(int query_position) const {
    switch (kind) {
    case CpuAttentionPatternKind::SlidingWindow:
        return std::max(0, query_position - window + 1);
    case CpuAttentionPatternKind::BlockSparse:
        return 0;
    case CpuAttentionPatternKind::DynamicSparse:
        return 0;
    default:
        return 0;
    }
}

CpuAttentionBias CpuAttentionBias::lower(const AttentionBiasSpec& bias,
                                         std::span<const float> relative_values,
                                         int query_heads) {
    if (const auto* alibi = std::get_if<AlibiBiasSpec>(&bias)) {
        return {alibi->slopes.data(), alibi->slopes.size()};
    }
    if (const auto* relative = std::get_if<RelativePositionBiasSpec>(&bias)) {
        if (query_heads <= 0 || relative->bucket_count <= 0 ||
            relative->max_distance <= 0 || relative_values.size() !=
                static_cast<size_t>(query_heads) *
                    static_cast<size_t>(relative->bucket_count)) {
            throw std::invalid_argument("relative position bias dimensions are invalid");
        }
        return {nullptr, 0, relative_values.data(), relative->bucket_count,
                relative->max_distance, relative->bidirectional};
    }
    return {};
}

float CpuAttentionBias::score(int query_head, int query_position,
                              int key_position) const {
    if (empty()) return 0.0f;
    if (relative_values != nullptr && relative_bucket_count != 0) {
        if (query_head < 0) {
            throw std::invalid_argument("relative position bias query head is out of range");
        }
        const int relative_position = key_position - query_position;
        const int bucket_count = relative_bidirectional
            ? relative_bucket_count / 2 : relative_bucket_count;
        if (bucket_count <= 0) {
            throw std::invalid_argument("relative position bias bucket count is invalid");
        }
        const bool positive = relative_bidirectional && relative_position > 0;
        const int distance = relative_bidirectional
            ? std::abs(relative_position) : std::max(-relative_position, 0);
        const int max_exact = bucket_count / 2;
        int bucket = 0;
        if (distance < max_exact) {
            bucket = distance;
        } else {
            const float denominator = std::log(
                static_cast<float>(std::max(relative_max_distance, max_exact + 1)) /
                static_cast<float>(std::max(max_exact, 1)));
            const float logarithmic = denominator == 0.0f ? 0.0f : std::log(
                static_cast<float>(std::max(distance, max_exact)) /
                static_cast<float>(std::max(max_exact, 1))) / denominator;
            bucket = max_exact + static_cast<int>(
                logarithmic * static_cast<float>(bucket_count - max_exact));
            bucket = std::min(bucket, bucket_count - 1);
        }
        if (positive) bucket += bucket_count;
        return relative_values[static_cast<size_t>(query_head) *
                               static_cast<size_t>(relative_bucket_count) +
                               static_cast<size_t>(bucket)];
    }
    if (query_head < 0 || static_cast<size_t>(query_head) >= slope_count) {
        throw std::invalid_argument("ALiBi query head is out of range");
    }
    return -alibi_slopes[static_cast<size_t>(query_head)] *
        static_cast<float>(std::abs(query_position - key_position));
}

} // namespace celeg
