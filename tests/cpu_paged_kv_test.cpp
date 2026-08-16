#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

static std::vector<float> reference_pattern(
    const std::vector<float>& query, const std::vector<float>& keys,
    const std::vector<float>& values, int sequence, int q_heads, int kv_heads,
    int head_dim, int query_position, const celeg::CpuAttentionPattern& pattern,
    const celeg::CpuAttentionBias& bias = {}) {
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> output(static_cast<size_t>(q_heads) * head_dim, 0.0f);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        std::vector<float> scores;
        scores.reserve(sequence);
        for (int token = 0; token < sequence; ++token) {
            if (!pattern.allows(query_position, token)) continue;
            float score = 0.0f;
            const float* key = keys.data() +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            const float* q = query.data() + static_cast<size_t>(qh) * head_dim;
            for (int d = 0; d < head_dim; ++d) score += q[d] * key[d];
            scores.push_back(score * scale + bias.score(qh, query_position, token));
        }
        const float maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0f;
        for (float score : scores) denominator += std::exp(score - maximum);
        float* destination = output.data() + static_cast<size_t>(qh) * head_dim;
        int score_index = 0;
        for (int token = 0; token < sequence; ++token) {
            if (!pattern.allows(query_position, token)) continue;
            const float weight = std::exp(scores[static_cast<size_t>(score_index++)] - maximum) /
                denominator;
            const float* value = values.data() +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) destination[d] += weight * value[d];
        }
    }
    return output;
}

static void run(celeg::CpuKvCacheMode mode) {
    constexpr int sequence = 11;
    constexpr int page_tokens = 4;
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 8;
    constexpr int kv_width = kv_heads * head_dim;

    celeg::CpuKvPagePool pool(mode, page_tokens,
                              celeg::CpuStatePageLayout{kv_width, kv_width, 0, 0});
    std::vector<celeg::CpuKvPageId> pages;
    std::vector<float> keys(sequence * kv_width);
    std::vector<float> values(sequence * kv_width);
    std::vector<float> query(q_heads * head_dim);
    std::mt19937 rng(19);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : keys) x = dist(rng);
    for (float& x : values) x = dist(rng);
    for (float& x : query) x = dist(rng);

    std::vector<float> reference_keys = keys;
    std::vector<float> reference_values = values;
    if (mode == celeg::CpuKvCacheMode::Bf16) {
        for (float& x : reference_keys) x = celeg::bf16_bits_to_float(celeg::float_to_bf16_bits(x));
        for (float& x : reference_values) x = celeg::bf16_bits_to_float(celeg::float_to_bf16_bits(x));
    }

    for (int token = 0; token < sequence; ++token) {
        if (token % page_tokens == 0) pages.push_back(pool.allocate());
        pool.write(pages.back(), token % page_tokens,
                   keys.data() + token * kv_width,
                   values.data() + token * kv_width);
    }
    std::vector<float> expected(q_heads * head_dim);
    std::vector<float> actual(q_heads * head_dim);
    celeg::cpu_gqa_decode(query.data(), reference_keys.data(), reference_values.data(),
                        expected.data(), sequence, q_heads, kv_heads, head_dim);
    celeg::cpu_gqa_decode_paged(query.data(), pool, pages, actual.data(),
                              sequence, q_heads, kv_heads, head_dim);
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    CELEG_TEST_CHECK(max_error < 1e-5f);

    if (mode == celeg::CpuKvCacheMode::Fp32) {
        const std::array<std::pair<celeg::AttentionPatternSpec, int>, 4> patterns = {{
            {celeg::BidirectionalPattern{}, 5},
            {celeg::PrefixLmPattern{4}, 2},
            {celeg::BlockSparsePattern{4, 1, 1}, 9},
            {celeg::DynamicSparsePattern{4, 2}, 9},
        }};
        for (const auto& [semantic, query_position] : patterns) {
            const auto lowered = celeg::CpuAttentionPattern::lower(semantic);
            const auto expected_pattern = reference_pattern(
                query, reference_keys, reference_values, sequence,
                q_heads, kv_heads, head_dim, query_position, lowered);
            std::vector<float> actual_pattern(query.size());
            celeg::cpu_gqa_decode_paged(
                query.data(), pool, pages, actual_pattern.data(), sequence,
                q_heads, kv_heads, head_dim, lowered, {}, query_position);
            for (size_t i = 0; i < actual_pattern.size(); ++i) {
                CELEG_TEST_CHECK(std::abs(actual_pattern[i] - expected_pattern[i]) < 1e-5f);
            }
        }

        const std::vector<float> slopes(q_heads, 0.05f);
        const celeg::CpuAttentionBias alibi{
            celeg::CpuAlibiBiasView{slopes.data(), slopes.size()}};
        const auto expected_alibi = reference_pattern(
            query, reference_keys, reference_values, sequence,
            q_heads, kv_heads, head_dim, sequence - 1,
            celeg::CpuAttentionPattern{}, alibi);
        std::vector<float> actual_alibi(query.size());
        celeg::cpu_gqa_decode_paged(
            query.data(), pool, pages, actual_alibi.data(), sequence,
            q_heads, kv_heads, head_dim, {}, alibi);
        for (size_t i = 0; i < actual_alibi.size(); ++i) {
            CELEG_TEST_CHECK(std::abs(actual_alibi[i] - expected_alibi[i]) < 1e-5f);
        }

        const celeg::RelativePositionBiasSpec relative_spec{8, 16, true};
        std::vector<float> relative_values(q_heads * relative_spec.bucket_count);
        for (int head = 0; head < q_heads; ++head) {
            for (int bucket = 0; bucket < relative_spec.bucket_count; ++bucket) {
                relative_values[head * relative_spec.bucket_count + bucket] =
                    static_cast<float>(head * 100 + bucket);
            }
        }
        const auto relative = celeg::CpuAttentionBias::lower(
            relative_spec, relative_values, q_heads);
        CELEG_TEST_CHECK(relative.score(0, 4, 4) == 0.0f);
        CELEG_TEST_CHECK(relative.score(0, 4, 5) == 5.0f);
        CELEG_TEST_CHECK(relative.score(0, 4, 3) == 1.0f);
        CELEG_TEST_CHECK(relative.score(2, 4, 5) == 205.0f);
        const auto expected_relative = reference_pattern(
            query, reference_keys, reference_values, sequence,
            q_heads, kv_heads, head_dim, sequence - 1,
            celeg::CpuAttentionPattern{}, relative);
        std::vector<float> actual_relative(query.size());
        celeg::cpu_gqa_decode_paged(
            query.data(), pool, pages, actual_relative.data(), sequence,
            q_heads, kv_heads, head_dim, {}, relative);
        for (size_t i = 0; i < actual_relative.size(); ++i) {
            CELEG_TEST_CHECK(std::abs(actual_relative[i] - expected_relative[i]) < 1e-5f);
        }

        celeg::CpuExternalAttentionMemory external_memory(
            celeg::CpuKvCacheMode::Bf16, 2, kv_width, kv_width);
        for (int token = 0; token < sequence; ++token) {
            external_memory.append(keys.data() + token * kv_width,
                                   values.data() + token * kv_width);
        }
        std::vector<float> external_output(query.size());
        celeg::cpu_gqa_decode_paged(
            query.data(), external_memory.pool(), external_memory.pages(),
            external_output.data(), static_cast<int>(external_memory.token_count()),
            q_heads, kv_heads, head_dim,
            celeg::CpuAttentionPattern::lower(celeg::BidirectionalPattern{}), {},
            sequence - 1);
        const auto external_expected = reference_pattern(
            query, reference_keys, reference_values, sequence,
            q_heads, kv_heads, head_dim, sequence - 1,
            celeg::CpuAttentionPattern::lower(celeg::BidirectionalPattern{}));
        for (size_t i = 0; i < external_output.size(); ++i) {
            CELEG_TEST_CHECK(std::abs(external_output[i] - external_expected[i]) < 1e-3f);
        }
        external_memory.clear();
        CELEG_TEST_CHECK(external_memory.token_count() == 0);
    }
    CELEG_TEST_CHECK(pool.stats().used_pages == pages.size());
    pool.retain(pages.front());
    CELEG_TEST_CHECK(pool.reference_count(pages.front()) == 2);
    pool.release(pages.front());
    for (auto page : pages) pool.release(page);
    CELEG_TEST_CHECK(pool.stats().used_pages == 0);
}

int main() {
    run(celeg::CpuKvCacheMode::Fp32);
    run(celeg::CpuKvCacheMode::Bf16);
    celeg::CpuKvPagePool latent_pool(
        celeg::CpuKvCacheMode::Bf16, 4,
        celeg::CpuStatePageLayout{0, 0, 8, 4});
    const auto latent_page = latent_pool.allocate();
    CELEG_TEST_CHECK(latent_pool.page_bytes() == 4 * 12 * sizeof(uint16_t));
    latent_pool.release(latent_page);
    celeg::CpuKvPagePool mixed_pool(
        celeg::CpuKvCacheMode::Fp32, 4,
        celeg::CpuStatePageLayout{2, 3, 4, 1});
    const auto mixed_page = mixed_pool.allocate();
    const float mixed_key[] = {1.0f, 2.0f};
    const float mixed_value[] = {3.0f, 4.0f, 5.0f};
    mixed_pool.write(mixed_page, 0, mixed_key, mixed_value);
    const auto mixed_clone = mixed_pool.clone_prefix(mixed_page, 1);
    CELEG_TEST_CHECK(mixed_pool.key_fp32(mixed_clone, 0)[1] == 2.0f);
    CELEG_TEST_CHECK(mixed_pool.value_fp32(mixed_clone, 0)[2] == 5.0f);
    mixed_pool.release(mixed_clone);
    mixed_pool.release(mixed_page);

    constexpr int latent_sequence = 5;
    constexpr int latent_heads = 2;
    constexpr int latent_rank = 3;
    constexpr int latent_rope = 1;
    celeg::CpuKvPagePool latent_attention_pool(
        celeg::CpuKvCacheMode::Fp32, 2,
        celeg::CpuStatePageLayout{0, 0, 2 * latent_rank,
                                  latent_rope});
    std::vector<celeg::CpuKvPageId> latent_pages;
    std::vector<float> latent_keys(latent_sequence * latent_rank);
    std::vector<float> latent_values(latent_sequence * latent_rank);
    std::vector<float> latent_key_rope(latent_sequence * latent_heads * latent_rope);
    std::vector<float> latent_query(latent_heads * latent_rank, 0.0f);
    std::vector<float> latent_query_rope(latent_heads * latent_rope, 0.0f);
    for (int token = 0; token < latent_sequence; ++token) {
        if (token % 2 == 0) latent_pages.push_back(latent_attention_pool.allocate());
        for (int d = 0; d < latent_rank; ++d) {
            latent_keys[token * latent_rank + d] = 0.1f * (token + d + 1);
            latent_values[token * latent_rank + d] = 0.2f * (token + 2 * d + 1);
        }
        for (int h = 0; h < latent_heads; ++h) {
            latent_key_rope[token * latent_heads + h] = 0.03f * (token + h + 1);
        }
        latent_attention_pool.write_latent(
            latent_pages.back(), token % 2,
            latent_keys.data() + token * latent_rank,
            latent_values.data() + token * latent_rank,
            latent_key_rope.data() + token * latent_heads);
    }
    std::vector<float> latent_output(latent_heads * latent_rank);
    celeg::cpu_latent_attention_decode_paged(
        latent_query.data(), latent_query_rope.data(), latent_attention_pool,
        latent_pages, latent_output.data(), latent_sequence, latent_heads,
        latent_rank, latent_rope, 1.0f);
    for (int h = 0; h < latent_heads; ++h) {
        for (int d = 0; d < latent_rank; ++d) {
            float expected_value = 0.0f;
            for (int token = 0; token < latent_sequence; ++token) {
                expected_value += latent_values[token * latent_rank + d] /
                    static_cast<float>(latent_sequence);
            }
            CELEG_TEST_CHECK(std::abs(latent_output[h * latent_rank + d] - expected_value) < 1e-6f);
        }
    }
    for (auto page : latent_pages) latent_attention_pool.release(page);

    std::cout << "cpu_paged_kv_test: ok\n";
}
