#include "celeg/backend/cpu/attention.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace celeg {
namespace {

inline float attention_value(float value) {
    return value;
}

inline float attention_value(uint16_t value) {
    return bf16_bits_to_float(value);
}

template <typename CacheT>
void cpu_gqa_decode_core(const float* q, const CacheT* key_cache,
                         const CacheT* value_cache, float* output,
                         int sequence_length, int q_heads, int kv_heads,
                         int head_dim) {
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    thread_local std::vector<float> scores;
    scores.resize(static_cast<size_t>(sequence_length));
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < sequence_length; ++token) {
            const CacheT* key = key_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += query[d] * attention_value(key[d]);
            }
            score *= scale;
            scores[static_cast<size_t>(token)] = score;
            maximum = std::max(maximum, score);
        }
        float denominator = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        float* destination = output + static_cast<size_t>(qh) * head_dim;
        std::fill(destination, destination + head_dim, 0.0f);
        const float inv = 1.0f / denominator;
        for (int token = 0; token < sequence_length; ++token) {
            const float probability = scores[static_cast<size_t>(token)] * inv;
            const CacheT* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                destination[d] += probability * attention_value(value[d]);
            }
        }
    }
}

}

void cpu_gqa_decode(const float* q, const float* key_cache,
                    const float* value_cache, float* output,
                    int sequence_length, int q_heads, int kv_heads,
                    int head_dim) {
    if (!q || !key_cache || !value_cache || !output || sequence_length <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid GQA arguments");
    }
    cpu_gqa_decode_core(q, key_cache, value_cache, output, sequence_length,
                        q_heads, kv_heads, head_dim);
}

void cpu_gqa_decode_bf16(const float* q, const uint16_t* key_cache,
                         const uint16_t* value_cache, float* output,
                         int sequence_length, int q_heads, int kv_heads,
                         int head_dim) {
    if (!q || !key_cache || !value_cache || !output || sequence_length <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid BF16 GQA arguments");
    }
    cpu_gqa_decode_core(q, key_cache, value_cache, output, sequence_length,
                        q_heads, kv_heads, head_dim);
}

}
