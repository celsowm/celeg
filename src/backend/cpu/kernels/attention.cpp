#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace celeg {

void cpu_gqa_decode(const float* q, const float* key_cache,
                    const float* value_cache, float* output,
                    int sequence_length, int q_heads, int kv_heads,
                    int head_dim) {
    if (!q || !key_cache || !value_cache || !output || sequence_length <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid GQA arguments");
    }
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    thread_local std::vector<float> scores;
    scores.resize(static_cast<size_t>(sequence_length));
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < sequence_length; ++token) {
            const float* key = key_cache + (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += query[d] * key[d];
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
            const float* value = value_cache + (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) destination[d] += probability * value[d];
        }
    }
}

void cpu_gqa_decode_bf16(const float* q, const uint16_t* key_cache,
                         const uint16_t* value_cache, float* output,
                         int sequence_length, int q_heads, int kv_heads,
                         int head_dim) {
    if (!q || !key_cache || !value_cache || !output || sequence_length <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid BF16 GQA arguments");
    }
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    thread_local std::vector<float> scores;
    scores.resize(static_cast<size_t>(sequence_length));
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < sequence_length; ++token) {
            const uint16_t* key = key_cache + (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += query[d] * bf16_bits_to_float(key[d]);
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
            const uint16_t* value = value_cache + (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) destination[d] += probability * bf16_bits_to_float(value[d]);
        }
    }
}

} // namespace celeg
