#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/attention/dynamic_sparse_semantics.hpp"
#include "celeg/attention/micro_semantics.hpp"
#include "celeg/attention/online_semantics.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace celeg {

namespace {

float dynamic_sparse_key_dot(const float* query,
                             const CpuKvPagePool& pool,
                             std::span<const CpuKvPageId> pages,
                             int token,
                             int kv_head,
                             int head_dim) {
    const size_t page_index = static_cast<size_t>(token) / pool.page_tokens();
    const size_t token_offset = static_cast<size_t>(token) % pool.page_tokens();
    float dot = 0.0f;
    if (pool.mode() == CpuKvCacheMode::Fp32) {
        const float* key = pool.key_fp32(pages[page_index], token_offset) +
            static_cast<size_t>(kv_head) * head_dim;
        for (int d = 0; d < head_dim; ++d) dot += query[d] * key[d];
    } else {
        const uint16_t* key = pool.key_bf16(pages[page_index], token_offset) +
            static_cast<size_t>(kv_head) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * bf16_bits_to_float(key[d]);
        }
    }
    return dot;
}

void dynamic_sparse_accumulate_value(float* destination,
                                     const CpuKvPagePool& pool,
                                     std::span<const CpuKvPageId> pages,
                                     int token,
                                     int kv_head,
                                     int head_dim,
                                     const attention_semantics::OnlineTransition& transition) {
    const size_t page_index = static_cast<size_t>(token) / pool.page_tokens();
    const size_t token_offset = static_cast<size_t>(token) % pool.page_tokens();
    if (pool.mode() == CpuKvCacheMode::Fp32) {
        const float* value = pool.value_fp32(pages[page_index], token_offset) +
            static_cast<size_t>(kv_head) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            destination[d] = attention_semantics::online_accumulate(
                destination[d], value[d], transition);
        }
    } else {
        const uint16_t* value = pool.value_bf16(pages[page_index], token_offset) +
            static_cast<size_t>(kv_head) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            destination[d] = attention_semantics::online_accumulate(
                destination[d], bf16_bits_to_float(value[d]), transition);
        }
    }
}

void validate_dynamic_sparse_pattern(const DynamicSparsePattern& pattern) {
    if (pattern.block_size <= 0 || pattern.max_selected_blocks <= 0) {
        throw std::invalid_argument(
            "CPU dynamic sparse attention requires positive block geometry");
    }
}

}

std::vector<int> cpu_dynamic_sparse_select_blocks(
    const float* query,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    int sequence_length,
    int kv_head,
    int head_dim,
    int query_position,
    const DynamicSparsePattern& pattern) {
    if (!query || sequence_length <= 0 || kv_head < 0 || head_dim <= 0 ||
        query_position < 0 || query_position >= sequence_length) {
        throw std::invalid_argument("invalid CPU dynamic sparse selection arguments");
    }
    validate_dynamic_sparse_pattern(pattern);
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    if (pages.size() < required_pages) {
        throw std::invalid_argument(
            "CPU dynamic sparse page table is incomplete");
    }
    if (pool.key_width() < static_cast<size_t>((kv_head + 1) * head_dim)) {
        throw std::invalid_argument("CPU dynamic sparse KV width mismatch");
    }

    const int candidate_count = attention_semantics::dynamic_sparse_candidate_count(
        query_position, pattern.block_size);
    const int selected_capacity = std::min(
        pattern.max_selected_blocks, candidate_count);
    const float scale = attention_semantics::attention_scale(head_dim);
    std::vector<float> candidate_scores(
        static_cast<size_t>(candidate_count),
        attention_semantics::dynamic_sparse_lowest_score());

    for (int block = 0; block < candidate_count; ++block) {
        const int begin = attention_semantics::dynamic_sparse_block_begin(
            block, pattern.block_size);
        const int end = attention_semantics::dynamic_sparse_block_end_exclusive(
            query_position, block, pattern.block_size);
        float maximum = attention_semantics::dynamic_sparse_lowest_score();
        for (int token = begin; token < end; ++token) {
            maximum = std::max(
                maximum,
                dynamic_sparse_key_dot(
                    query, pool, pages, token, kv_head, head_dim) * scale);
        }
        candidate_scores[static_cast<size_t>(block)] = maximum;
    }

    std::vector<int> selected(static_cast<size_t>(selected_capacity), -1);
    std::vector<float> selected_scores(
        static_cast<size_t>(selected_capacity),
        attention_semantics::dynamic_sparse_lowest_score());
    attention_semantics::dynamic_sparse_select_top_k(
        candidate_scores.data(), candidate_count, selected_capacity,
        selected.data(), selected_scores.data());
    selected.erase(
        std::remove(selected.begin(), selected.end(), -1), selected.end());
    return selected;
}

void cpu_gqa_decode_paged_dynamic_sparse(
    const float* q,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages,
    float* output,
    int sequence_length,
    int q_heads,
    int kv_heads,
    int head_dim,
    const DynamicSparsePattern& pattern,
    CpuAttentionBias bias,
    int query_position) {
    if (!q || !output || sequence_length <= 0 || q_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0 || q_heads % kv_heads != 0) {
        throw std::invalid_argument("invalid CPU dynamic sparse attention dimensions");
    }
    validate_dynamic_sparse_pattern(pattern);
    if (query_position < 0) {
        query_position = attention_semantics::query_position_from_sequence_length(
            sequence_length);
    }
    if (query_position >= sequence_length) {
        throw std::invalid_argument(
            "CPU dynamic sparse query position is out of range");
    }
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    if (pages.size() < required_pages) {
        throw std::invalid_argument(
            "CPU dynamic sparse page table is incomplete");
    }
    const size_t kv_width = static_cast<size_t>(kv_heads * head_dim);
    if (pool.key_width() != kv_width || pool.value_width() != kv_width) {
        throw std::invalid_argument("CPU dynamic sparse KV width mismatch");
    }

    const float scale = attention_semantics::attention_scale(head_dim);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = attention_semantics::gqa_kv_head(
            qh, q_heads, kv_heads);
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float* destination = output + static_cast<size_t>(qh) * head_dim;
        std::fill(destination, destination + head_dim, 0.0f);

        const std::vector<int> selected = cpu_dynamic_sparse_select_blocks(
            query, pool, pages, sequence_length, kvh, head_dim,
            query_position, pattern);
        if (selected.empty()) {
            throw std::invalid_argument(
                "CPU dynamic sparse attention selected no blocks");
        }

        float maximum = -std::numeric_limits<float>::infinity();
        float denominator = 0.0f;
        for (int token = 0; token <= query_position; ++token) {
            const int block = token / pattern.block_size;
            if (!attention_semantics::dynamic_sparse_selected_block(
                    block, selected.data(), static_cast<int>(selected.size()))) {
                continue;
            }
            const float score = dynamic_sparse_key_dot(
                query, pool, pages, token, kvh, head_dim) * scale +
                bias.score(qh, query_position, token);
            const auto transition = attention_semantics::online_transition(
                maximum, denominator, score);
            denominator = transition.denominator;
            dynamic_sparse_accumulate_value(
                destination, pool, pages, token, kvh, head_dim, transition);
            maximum = transition.maximum;
        }
        if (denominator == 0.0f) {
            throw std::invalid_argument(
                "CPU dynamic sparse attention selected no KV tokens");
        }
        const float reciprocal = 1.0f / denominator;
        for (int d = 0; d < head_dim; ++d) destination[d] *= reciprocal;
    }
}

}
