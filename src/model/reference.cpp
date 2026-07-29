#include "lfm/model/reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lfm::reference {

uint16_t float_to_bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t exponent = bits & 0x7f800000u;
    const uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0) {
        // Preserve NaN and force a non-zero BF16 payload.
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

float round_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
}

std::vector<float> rmsnorm_bf16(const std::vector<float>& x,
                                const std::vector<float>& weight,
                                float eps) {
    if (x.size() != weight.size() || x.empty()) {
        throw std::invalid_argument("rmsnorm_bf16 shape mismatch");
    }
    float sum = 0.0f;
    for (float value : x) {
        const float v = round_bf16(value);
        sum += v * v;
    }
    const float inv = 1.0f / std::sqrt(sum / static_cast<float>(x.size()) + eps);
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        // Transformers casts normalized states back to the input dtype before
        // multiplying by the BF16 norm weight.
        const float normalized = round_bf16(round_bf16(x[i]) * inv);
        out[i] = round_bf16(normalized * round_bf16(weight[i]));
    }
    return out;
}

void rope_bf16_inplace(std::vector<float>& vector,
                       const std::vector<float>& cos_half,
                       const std::vector<float>& sin_half) {
    if ((vector.size() % 2) != 0 || cos_half.size() * 2 != vector.size() ||
        sin_half.size() != cos_half.size()) {
        throw std::invalid_argument("rope_bf16_inplace shape mismatch");
    }
    const size_t half = vector.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        const float a = round_bf16(vector[i]);
        const float b = round_bf16(vector[i + half]);
        const float c = round_bf16(cos_half[i]);
        const float s = round_bf16(sin_half[i]);
        const float ac = round_bf16(a * c);
        const float bs = round_bf16(b * s);
        const float bc = round_bf16(b * c);
        const float as = round_bf16(a * s);
        vector[i] = round_bf16(ac - bs);
        vector[i + half] = round_bf16(bc + as);
    }
}

std::vector<float> gqa_decode_strict_bf16(
    const std::vector<float>& q,
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim) {
    if (seq_len <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_heads % kv_heads != 0 ||
        q.size() != static_cast<size_t>(q_heads * head_dim) ||
        key_cache.size() != static_cast<size_t>(seq_len * kv_heads * head_dim) ||
        value_cache.size() != key_cache.size()) {
        throw std::invalid_argument("gqa_decode_strict_bf16 shape mismatch");
    }

    std::vector<float> out(static_cast<size_t>(q_heads * head_dim));
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int qh = 0; qh < q_heads; ++qh) {
        const int kh = qh / queries_per_kv;
        std::vector<float> scores(static_cast<size_t>(seq_len));
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < seq_len; ++token) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                const float qv = round_bf16(q[static_cast<size_t>(qh * head_dim + d)]);
                const size_t index =
                    (static_cast<size_t>(token) * kv_heads + kh) * head_dim + d;
                dot += qv * round_bf16(key_cache[index]);
            }
            // BF16 matmul produces a BF16 score before the scalar scale;
            // the BF16 scalar multiplication then rounds once more.
            scores[static_cast<size_t>(token)] =
                round_bf16(round_bf16(dot) * scale);
            maximum = std::max(maximum, scores[static_cast<size_t>(token)]);
        }

        float denominator = 0.0f;
        for (float score : scores) denominator += std::exp(score - maximum);

        for (int d = 0; d < head_dim; ++d) {
            float accumulator = 0.0f;
            for (int token = 0; token < seq_len; ++token) {
                const float probability = round_bf16(
                    std::exp(scores[static_cast<size_t>(token)] - maximum) / denominator);
                const size_t index =
                    (static_cast<size_t>(token) * kv_heads + kh) * head_dim + d;
                accumulator += probability * round_bf16(value_cache[index]);
            }
            out[static_cast<size_t>(qh * head_dim + d)] = round_bf16(accumulator);
        }
    }
    return out;
}

std::vector<float> conv_decode_bf16(
    const std::vector<float>& projected_bcx,
    const std::vector<float>& conv_weight,
    std::vector<float>& state,
    int hidden,
    int cache_length,
    int position) {
    if (hidden <= 0 || cache_length <= 0 || position < 0 ||
        projected_bcx.size() != static_cast<size_t>(3 * hidden) ||
        conv_weight.size() != static_cast<size_t>(hidden * cache_length) ||
        state.size() != static_cast<size_t>(hidden * cache_length)) {
        throw std::invalid_argument("conv_decode_bf16 shape mismatch");
    }

    const int cursor = position % cache_length;
    std::vector<float> out(static_cast<size_t>(hidden));
    for (int channel = 0; channel < hidden; ++channel) {
        const float b = round_bf16(projected_bcx[static_cast<size_t>(channel)]);
        const float c = round_bf16(projected_bcx[static_cast<size_t>(hidden + channel)]);
        const float x = round_bf16(projected_bcx[static_cast<size_t>(2 * hidden + channel)]);
        state[static_cast<size_t>(cursor * hidden + channel)] = round_bf16(b * x);

        float conv = 0.0f;
        for (int tap = 0; tap < cache_length; ++tap) {
            const int slot = (cursor + 1 + tap) % cache_length;
            conv += round_bf16(state[static_cast<size_t>(slot * hidden + channel)]) *
                    round_bf16(conv_weight[static_cast<size_t>(channel * cache_length + tap)]);
        }
        const float conv_bf16 = round_bf16(conv);
        out[static_cast<size_t>(channel)] = round_bf16(c * conv_bf16);
    }
    return out;
}

} // namespace lfm::reference
