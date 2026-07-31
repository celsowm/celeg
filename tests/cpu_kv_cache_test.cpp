#include "celeg/backend/cpu/kernels.hpp"
#include "support/assertions.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr int sequence = 7;
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 8;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> q(q_heads * head_dim);
    std::vector<float> key(sequence * kv_heads * head_dim);
    std::vector<float> value(sequence * kv_heads * head_dim);
    for (float& x : q) x = dist(rng);
    for (float& x : key) x = dist(rng);
    for (float& x : value) x = dist(rng);

    std::vector<uint16_t> key_bf16(key.size());
    std::vector<uint16_t> value_bf16(value.size());
    std::vector<float> key_rounded(key.size());
    std::vector<float> value_rounded(value.size());
    for (size_t i = 0; i < key.size(); ++i) {
        key_bf16[i] = celeg::float_to_bf16_bits(key[i]);
        key_rounded[i] = celeg::bf16_bits_to_float(key_bf16[i]);
    }
    for (size_t i = 0; i < value.size(); ++i) {
        value_bf16[i] = celeg::float_to_bf16_bits(value[i]);
        value_rounded[i] = celeg::bf16_bits_to_float(value_bf16[i]);
    }

    std::vector<float> reference(q_heads * head_dim);
    std::vector<float> actual(q_heads * head_dim);
    celeg::cpu_gqa_decode(q.data(), key_rounded.data(), value_rounded.data(),
                        reference.data(), sequence, q_heads, kv_heads, head_dim);
    celeg::cpu_gqa_decode_bf16(q.data(), key_bf16.data(), value_bf16.data(),
                             actual.data(), sequence, q_heads, kv_heads, head_dim);

    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(actual[i] - reference[i]));
    }
    CELEG_TEST_CHECK(maximum < 1e-6f);
    std::cout << "cpu_kv_cache_test: ok max_error=" << maximum << '\n';
    return 0;
}
