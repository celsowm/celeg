#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct OracleResult {
    std::vector<int> selected;
    std::vector<float> output;
};

float rounded_for_mode(float value, celeg::CpuKvCacheMode mode) {
    if (mode == celeg::CpuKvCacheMode::Fp32) return value;
    return celeg::bf16_bits_to_float(celeg::float_to_bf16_bits(value));
}

OracleResult dense_dynamic_sparse_oracle(
    const float* query,
    const std::vector<float>& keys,
    const std::vector<float>& values,
    int sequence_length,
    int kv_head,
    int kv_heads,
    int head_dim,
    int query_position,
    const celeg::DynamicSparsePattern& pattern,
    celeg::CpuKvCacheMode mode,
    int query_head = 0,
    const celeg::CpuAttentionBias& bias = {}) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int candidate_count = query_position / pattern.block_size + 1;
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(candidate_count));
    for (int block = 0; block < candidate_count; ++block) {
        const int begin = block * pattern.block_size;
        const int end = std::min((block + 1) * pattern.block_size,
                                 query_position + 1);
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = begin; token < end; ++token) {
            const float* key = keys.data() +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * rounded_for_mode(key[d], mode);
            }
            maximum = std::max(maximum, dot * scale);
        }
        ranked.emplace_back(maximum, block);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        if (left.first != right.first) return left.first > right.first;
        return left.second < right.second;
    });
    const int selected_count = std::min(
        pattern.max_selected_blocks, candidate_count);
    OracleResult result;
    result.selected.reserve(static_cast<size_t>(selected_count));
    for (int index = 0; index < selected_count; ++index) {
        result.selected.push_back(ranked[static_cast<size_t>(index)].second);
    }

    std::vector<float> scores;
    std::vector<int> tokens;
    for (int token = 0; token <= query_position && token < sequence_length; ++token) {
        const int block = token / pattern.block_size;
        if (std::find(result.selected.begin(), result.selected.end(), block) ==
            result.selected.end()) {
            continue;
        }
        const float* key = keys.data() +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += query[d] * rounded_for_mode(key[d], mode);
        }
        scores.push_back(dot * scale +
                         bias.score(query_head, query_position, token));
        tokens.push_back(token);
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    float denominator = 0.0f;
    for (float score : scores) denominator += std::exp(score - maximum);
    result.output.assign(static_cast<size_t>(head_dim), 0.0f);
    for (size_t index = 0; index < tokens.size(); ++index) {
        const float probability = std::exp(scores[index] - maximum) / denominator;
        const float* value = values.data() +
            (static_cast<size_t>(tokens[index]) * kv_heads + kv_head) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            result.output[static_cast<size_t>(d)] +=
                probability * rounded_for_mode(value[d], mode);
        }
    }
    return result;
}

void check_close(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 float tolerance = 1.0e-5f) {
    CELEG_TEST_CHECK(actual.size() == expected.size());
    for (size_t index = 0; index < actual.size(); ++index) {
        CELEG_TEST_CHECK(std::abs(actual[index] - expected[index]) < tolerance);
    }
}

void run_mode(celeg::CpuKvCacheMode mode) {
    constexpr int sequence = 6;
    constexpr int page_tokens = 2;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    constexpr int kv_width = kv_heads * head_dim;

    const std::vector<float> keys = {
        -4.0f, 0.0f,
        -5.0f, 0.0f,
        -2.0f, 0.0f,
        -3.0f, 0.0f,
        -2.0f, 0.0f,
        100.0f, 0.0f,
    };
    const std::vector<float> values = {
        1.0f, 10.0f,
        2.0f, 20.0f,
        3.0f, 30.0f,
        4.0f, 40.0f,
        5.0f, 50.0f,
        6.0f, 60.0f,
    };
    const std::vector<float> query = {
        1.0f, 0.0f,
       -1.0f, 0.0f,
    };

    celeg::CpuKvPagePool pool(
        mode, page_tokens,
        celeg::CpuStatePageLayout{
            celeg::CpuOrdinaryKvPageLayout{kv_width, kv_width}});
    std::vector<celeg::CpuKvPageId> pages;
    for (int token = 0; token < sequence; ++token) {
        if (token % page_tokens == 0) pages.push_back(pool.allocate());
        pool.write(pages.back(), static_cast<size_t>(token % page_tokens),
                   keys.data() + static_cast<size_t>(token) * kv_width,
                   values.data() + static_cast<size_t>(token) * kv_width);
    }

    const celeg::DynamicSparsePattern top_one{2, 1};
    const auto partial_selected = celeg::cpu_dynamic_sparse_select_blocks(
        query.data(), pool, pages, sequence, 0, head_dim, 4, top_one);
    CELEG_TEST_CHECK(partial_selected.size() == 1);
    CELEG_TEST_CHECK(partial_selected[0] == 1);
    const auto partial_oracle = dense_dynamic_sparse_oracle(
        query.data(), keys, values, sequence, 0, kv_heads, head_dim, 4,
        top_one, mode);
    CELEG_TEST_CHECK(partial_oracle.selected == partial_selected);

    const auto tail_selected = celeg::cpu_dynamic_sparse_select_blocks(
        query.data(), pool, pages, sequence, 0, head_dim, 5, top_one);
    CELEG_TEST_CHECK(tail_selected.size() == 1);
    CELEG_TEST_CHECK(tail_selected[0] == 2);

    const celeg::DynamicSparsePattern top_two{2, 2};
    const auto tied_selected = celeg::cpu_dynamic_sparse_select_blocks(
        query.data(), pool, pages, sequence, 0, head_dim, 4, top_two);
    CELEG_TEST_CHECK(tied_selected.size() == 2);
    CELEG_TEST_CHECK(tied_selected[0] == 1);
    CELEG_TEST_CHECK(tied_selected[1] == 2);

    const celeg::CpuAttentionPattern lowered =
        celeg::CpuAttentionPattern::lower(top_one);
    CELEG_TEST_CHECK(!lowered.may_read_future(4, sequence));
    CELEG_TEST_CHECK(lowered.first_candidate(4) == 0);

    std::vector<float> actual(static_cast<size_t>(q_heads * head_dim));
    celeg::cpu_gqa_decode_paged(
        query.data(), pool, pages, actual.data(), sequence,
        q_heads, kv_heads, head_dim, lowered, {}, 4);
    std::vector<float> expected;
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / (q_heads / kv_heads);
        const auto oracle = dense_dynamic_sparse_oracle(
            query.data() + static_cast<size_t>(qh) * head_dim,
            keys, values, sequence, kvh, kv_heads, head_dim, 4,
            top_one, mode, qh);
        expected.insert(expected.end(), oracle.output.begin(), oracle.output.end());
    }
    check_close(actual, expected);

    std::vector<float> tail_actual(static_cast<size_t>(q_heads * head_dim));
    celeg::cpu_gqa_decode_paged(
        query.data(), pool, pages, tail_actual.data(), sequence,
        q_heads, kv_heads, head_dim, lowered, {}, 5);
    std::vector<float> tail_expected;
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / (q_heads / kv_heads);
        const auto oracle = dense_dynamic_sparse_oracle(
            query.data() + static_cast<size_t>(qh) * head_dim,
            keys, values, sequence, kvh, kv_heads, head_dim, 5,
            top_one, mode, qh);
        tail_expected.insert(
            tail_expected.end(), oracle.output.begin(), oracle.output.end());
    }
    check_close(tail_actual, tail_expected);

    const std::vector<float> prefill_queries = {
         1.0f, 0.0f, -1.0f, 0.0f,
         1.0f, 0.0f, -1.0f, 0.0f,
         1.0f, 0.0f, -1.0f, 0.0f,
    };
    std::vector<float> prefill_output(prefill_queries.size());
    celeg::CpuThreadPool thread_pool(2);
    celeg::cpu_gqa_prefill_paged(
        prefill_queries.data(), 3, q_heads * head_dim,
        pool, pages, prefill_output.data(), 3,
        q_heads, kv_heads, head_dim, thread_pool, lowered);
    for (int row = 0; row < 3; ++row) {
        const int position = 3 + row;
        for (int qh = 0; qh < q_heads; ++qh) {
            const int kvh = qh / (q_heads / kv_heads);
            const float* row_query = prefill_queries.data() +
                static_cast<size_t>(row * q_heads + qh) * head_dim;
            const auto oracle = dense_dynamic_sparse_oracle(
                row_query, keys, values, position + 1, kvh, kv_heads,
                head_dim, position, top_one, mode, qh);
            const float* row_output = prefill_output.data() +
                static_cast<size_t>(row * q_heads + qh) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                CELEG_TEST_CHECK(std::abs(
                    row_output[d] - oracle.output[static_cast<size_t>(d)]) < 1.0e-5f);
            }
        }
    }

    celeg::CpuPagedAttentionStats stats;
    std::vector<float> fallback_output(actual.size());
    celeg::cpu_gqa_decode_paged_parallel(
        query.data(), pool, pages, fallback_output.data(), sequence,
        q_heads, kv_heads, head_dim, thread_pool, lowered, {},
        celeg::CpuPagedAttentionOptions{1, 1}, &stats);
    CELEG_TEST_CHECK(!stats.parallel);
    check_close(fallback_output, tail_expected);

    bool rejected = false;
    try {
        (void)celeg::CpuAttentionPattern::lower(
            celeg::DynamicSparsePattern{0, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    for (celeg::CpuKvPageId page : pages) pool.release(page);
}

}

int main() {
    run_mode(celeg::CpuKvCacheMode::Fp32);
    run_mode(celeg::CpuKvCacheMode::Bf16);
    return 0;
}
