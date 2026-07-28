#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#include "gguf_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace lfm::detail {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define LFM_GGUF_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define LFM_GGUF_AVX2_TARGET
#endif

#pragma pack(push, 1)
struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
#pragma pack(pop)

float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign |
                static_cast<uint32_t>(127 - 15 - shift) << 23 |
                mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

LFM_GGUF_AVX2_TARGET int horizontal_sum(__m256i values) {
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

LFM_GGUF_AVX2_TARGET int dot_i8_u4(const int8_t* activation,
                                  const uint8_t* packed,
                                  bool high_nibble) {
    __m256i sum = _mm256_setzero_si256();
    const __m256i mask = _mm256_set1_epi32(0x0f);
    for (int i = 0; i < 32; i += 8) {
        const __m128i packed8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + i));
        __m256i q = _mm256_cvtepu8_epi32(packed8);
        q = high_nibble ? _mm256_srli_epi32(q, 4)
                        : _mm256_and_si256(q, mask);
        const __m128i activation8 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(activation + i));
        const __m256i x = _mm256_cvtepi8_epi32(activation8);
        sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(q, x));
    }
    return horizontal_sum(sum);
}

int q6_value(const BlockQ6K& block, int col) {
    const int half = col >> 7;
    const int index = col & 127;
    const int lane = index & 31;
    const int group = index >> 5;
    const uint8_t* ql = block.ql + half * 64;
    const uint8_t* qh = block.qh + half * 32;
    if (group == 0) return (ql[lane] & 15) | ((qh[lane] & 3) << 4);
    if (group == 1) return (ql[lane + 32] & 15) | (((qh[lane] >> 2) & 3) << 4);
    if (group == 2) return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

void scale_min(const BlockQ4K& block, int sub, uint8_t& scale,
               uint8_t& minimum) {
    if (sub < 4) {
        scale = block.scales[sub] & 63;
        minimum = block.scales[sub + 4] & 63;
    } else {
        scale = static_cast<uint8_t>(
            (block.scales[sub + 4] & 15) |
            ((block.scales[sub - 4] >> 6) << 4));
        minimum = static_cast<uint8_t>(
            (block.scales[sub + 4] >> 4) |
            ((block.scales[sub] >> 6) << 4));
    }
}

} // namespace

LFM_GGUF_AVX2_TARGET
void cpu_quantize_q8k_avx2(const float* input, size_t cols,
                           CpuQ8KBlock* output) {
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (size_t block_index = 0; block_index < cols / 256; ++block_index) {
        const float* source = input + block_index * 256;
        __m256 maximum = _mm256_setzero_ps();
        for (int i = 0; i < 256; i += 8) {
            const __m256 values = _mm256_loadu_ps(source + i);
            maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign_mask, values));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, maximum);
        float max_value = 0.0f;
        for (float lane : lanes) max_value = std::max(max_value, lane);
        CpuQ8KBlock& block = output[block_index];
        block.d = max_value == 0.0f ? 0.0f : max_value / 127.0f;
        const __m256 inverse = _mm256_set1_ps(
            block.d == 0.0f ? 0.0f : 1.0f / block.d);
        alignas(32) int32_t converted[8];
        for (int i = 0; i < 256; i += 8) {
            const __m256 values = _mm256_mul_ps(
                _mm256_loadu_ps(source + i), inverse);
            const __m256i integers = _mm256_cvtps_epi32(values);
            _mm256_store_si256(reinterpret_cast<__m256i*>(converted), integers);
            for (int lane = 0; lane < 8; ++lane) {
                block.qs[i + lane] = static_cast<int8_t>(
                    std::clamp(converted[lane], -127, 127));
            }
        }
        for (int group = 0; group < 16; ++group) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += block.qs[group * 16 + i];
            block.bsums[group] = static_cast<int16_t>(sum);
        }
    }
}

LFM_GGUF_AVX2_TARGET
float cpu_gguf_dot_avx2(const std::byte* packed_row, GgmlType type,
                        const CpuQ8KBlock* activation, size_t cols) {
    float total = 0.0f;
    const size_t blocks = cols / 256;
    if (type == GgmlType::Q4_K) {
        const auto* weights = reinterpret_cast<const BlockQ4K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ4K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const float d = fp16_to_float(weight.d);
            const float dmin = fp16_to_float(weight.dmin);
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                scale_min(weight, sub, scale, minimum);
                const int dot = dot_i8_u4(
                    x.qs.data() + sub * 32,
                    weight.qs + (sub >> 1) * 32, (sub & 1) != 0);
                const int sum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += d * static_cast<float>(scale * dot) -
                               dmin * static_cast<float>(minimum * sum);
            }
            total += x.d * block_total;
        }
        return total;
    }
    if (type == GgmlType::Q6_K) {
        const auto* weights = reinterpret_cast<const BlockQ6K*>(packed_row);
        alignas(16) int8_t decoded[16];
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ6K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            int block_total = 0;
            for (int sub = 0; sub < 16; ++sub) {
                for (int i = 0; i < 16; ++i) {
                    decoded[i] = static_cast<int8_t>(
                        q6_value(weight, sub * 16 + i) - 32);
                }
                __m256i dot = _mm256_setzero_si256();
                for (int i = 0; i < 16; i += 8) {
                    const __m128i q8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(decoded + i));
                    const __m128i x8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(
                            x.qs.data() + sub * 16 + i));
                    dot = _mm256_add_epi32(
                        dot, _mm256_mullo_epi32(
                            _mm256_cvtepi8_epi32(q8),
                            _mm256_cvtepi8_epi32(x8)));
                }
                block_total += static_cast<int>(weight.scales[sub]) *
                               horizontal_sum(dot);
            }
            total += fp16_to_float(weight.d) * x.d *
                     static_cast<float>(block_total);
        }
        return total;
    }
    throw std::invalid_argument("unsupported CPU GGUF AVX2 type");
}

#undef LFM_GGUF_AVX2_TARGET

} // namespace lfm::detail

#endif
