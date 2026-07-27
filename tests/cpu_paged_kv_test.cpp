#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/backend/cpu/paged_kv.hpp"
#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

static void run(lfm::CpuKvCacheMode mode) {
    constexpr int sequence = 11;
    constexpr int page_tokens = 4;
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 8;
    constexpr int kv_width = kv_heads * head_dim;

    lfm::CpuKvPagePool pool(mode, page_tokens, kv_width);
    std::vector<lfm::CpuKvPageId> pages;
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
    if (mode == lfm::CpuKvCacheMode::Bf16) {
        for (float& x : reference_keys) x = lfm::bf16_bits_to_float(lfm::float_to_bf16_bits(x));
        for (float& x : reference_values) x = lfm::bf16_bits_to_float(lfm::float_to_bf16_bits(x));
    }

    for (int token = 0; token < sequence; ++token) {
        if (token % page_tokens == 0) pages.push_back(pool.allocate());
        pool.write(pages.back(), token % page_tokens,
                   keys.data() + token * kv_width,
                   values.data() + token * kv_width);
    }
    std::vector<float> expected(q_heads * head_dim);
    std::vector<float> actual(q_heads * head_dim);
    lfm::cpu_gqa_decode(query.data(), reference_keys.data(), reference_values.data(),
                        expected.data(), sequence, q_heads, kv_heads, head_dim);
    lfm::cpu_gqa_decode_paged(query.data(), pool, pages, actual.data(),
                              sequence, q_heads, kv_heads, head_dim);
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    assert(max_error < 1e-5f);
    assert(pool.stats().used_pages == pages.size());
    pool.retain(pages.front());
    assert(pool.reference_count(pages.front()) == 2);
    pool.release(pages.front());
    for (auto page : pages) pool.release(page);
    assert(pool.stats().used_pages == 0);
}

int main() {
    run(lfm::CpuKvCacheMode::Fp32);
    run(lfm::CpuKvCacheMode::Bf16);
    std::cout << "cpu_paged_kv_test: ok\n";
}
