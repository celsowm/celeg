#include "lfm/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 8;
    constexpr int sequence = 37;
    lfm::CpuKvPagePool pool(lfm::CpuKvCacheMode::Bf16, 8,
                            kv_heads * head_dim);
    std::vector<lfm::CpuKvPageId> pages;
    std::vector<float> key(kv_heads * head_dim);
    std::vector<float> value(kv_heads * head_dim);
    for (int token = 0; token < sequence; ++token) {
        if (token % 8 == 0) pages.push_back(pool.allocate());
        for (size_t i = 0; i < key.size(); ++i) {
            key[i] = std::sin(static_cast<float>(token + 1) * (i + 1) * 0.013f);
            value[i] = std::cos(static_cast<float>(token + 3) * (i + 1) * 0.017f);
        }
        pool.write(pages.back(), static_cast<size_t>(token % 8),
                   key.data(), value.data());
    }
    std::vector<float> query(q_heads * head_dim);
    for (size_t i = 0; i < query.size(); ++i) query[i] = 0.01f * static_cast<float>(i + 1);
    std::vector<float> reference(query.size());
    std::vector<float> parallel(query.size());
    lfm::cpu_gqa_decode_paged(query.data(), pool, pages, reference.data(),
                              sequence, q_heads, kv_heads, head_dim);
    lfm::CpuThreadPool workers(4);
    lfm::CpuPagedAttentionStats stats;
    lfm::cpu_gqa_decode_paged_parallel(
        query.data(), pool, pages, parallel.data(), sequence,
        q_heads, kv_heads, head_dim, workers,
        lfm::CpuPagedAttentionOptions{1, 2}, &stats);
    LFM_TEST_CHECK(stats.parallel);
    LFM_TEST_CHECK(stats.tasks == q_heads * 3);
    float maximum = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        maximum = std::max(maximum, std::abs(reference[i] - parallel[i]));
    }
    LFM_TEST_CHECK(maximum < 1e-5f);

    // Chunk prefill must be equivalent to independently evaluated causal
    // queries, while using a single parallel region for the whole chunk.
    constexpr int base = 11;
    constexpr size_t query_rows = 5;
    std::vector<float> query_chunk(query_rows * query.size());
    std::vector<float> chunk_output(query_rows * query.size());
    std::vector<float> chunk_reference(query_rows * query.size());
    for (size_t row = 0; row < query_rows; ++row) {
        for (size_t i = 0; i < query.size(); ++i) {
            query_chunk[row * query.size() + i] = query[i] + 0.003f * static_cast<float>(row);
        }
        lfm::cpu_gqa_decode_paged(query_chunk.data() + row * query.size(), pool, pages,
                                  chunk_reference.data() + row * query.size(),
                                  base + static_cast<int>(row) + 1,
                                  q_heads, kv_heads, head_dim);
    }
    lfm::cpu_gqa_prefill_paged(query_chunk.data(), query_rows, query.size(), pool, pages,
                                chunk_output.data(), base, q_heads, kv_heads, head_dim,
                                workers);
    float chunk_maximum = 0.0f;
    for (size_t i = 0; i < chunk_output.size(); ++i) {
        chunk_maximum = std::max(chunk_maximum,
            std::abs(chunk_output[i] - chunk_reference[i]));
    }
    LFM_TEST_CHECK(chunk_maximum < 1e-5f);
    for (auto page : pages) pool.release(page);
    std::cout << "cpu_parallel_attention_test: ok max_error=" << maximum
              << " chunk_max_error=" << chunk_maximum << '\n';
}
