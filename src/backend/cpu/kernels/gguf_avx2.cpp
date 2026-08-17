#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)

#include "gguf_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace celeg::detail {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define CELEG_GGUF_AVX2_TARGET __attribute__((target("avx2,fma")))
#else
#define CELEG_GGUF_AVX2_TARGET
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
struct BlockQ4_0 {
    uint16_t d;
    uint8_t qs[16];
};
struct BlockQ5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};
#pragma pack(pop)

static_assert(sizeof(BlockQ4_0) == 18);
static_assert(sizeof(BlockQ5_0) == 22);

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
                static_cast<uint32_t>(127 - 14 - shift) << 23 |
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

CELEG_GGUF_AVX2_TARGET int horizontal_sum(__m256i values) {
    const __m128i low = _mm256_castsi256_si128(values);
    const __m128i high = _mm256_extracti128_si256(values, 1);
    __m128i sum = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

CELEG_GGUF_AVX2_TARGET int horizontal_sum_128(__m128i values) {
    values = _mm_hadd_epi32(values, values);
    values = _mm_hadd_epi32(values, values);
    return _mm_cvtsi128_si32(values);
}

CELEG_GGUF_AVX2_TARGET int dot_u8_i8_32(const uint8_t* weights,
                                      const int8_t* activation,
                                      bool high_nibble) {
    const __m256i packed = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(weights));
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i unpacked = high_nibble
        ? _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask)
        : _mm256_and_si256(packed, mask);
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation));
    const __m256i pair = _mm256_maddubs_epi16(unpacked, x);
    return horizontal_sum(_mm256_madd_epi16(pair, _mm256_set1_epi16(1)));
}

CELEG_GGUF_AVX2_TARGET int dot_u8_i8_16(const uint8_t* weights,
                                       const int8_t* activation) {
    const __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights));
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(activation));
    const __m128i pair = _mm_maddubs_epi16(w, x);
    return horizontal_sum_128(_mm_madd_epi16(pair, _mm_set1_epi16(1)));
}

CELEG_GGUF_AVX2_TARGET int dot_u8_i8_16_nib(const uint8_t* weights,
                                            const int8_t* activation,
                                            bool high_nibble) {
    const __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights));
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i unpacked = high_nibble
        ? _mm_and_si128(_mm_srli_epi16(w, 4), mask)
        : _mm_and_si128(w, mask);
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(activation));
    const __m128i pair = _mm_maddubs_epi16(unpacked, x);
    return horizontal_sum_128(_mm_madd_epi16(pair, _mm_set1_epi16(1)));
}

CELEG_GGUF_AVX2_TARGET __m128i expand8(uint8_t byte) {
    const __m128i src = _mm_cvtsi32_si128(static_cast<int>(byte));
    const __m128i bcast = _mm_shuffle_epi8(src, _mm_set1_epi8(0));
    const __m128i masks = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, -128,
                                        0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i anded = _mm_and_si128(bcast, masks);
    const __m128i bits = _mm_cmpgt_epi8(anded, _mm_setzero_si128());
    return _mm_and_si128(bits, _mm_set1_epi8(1));
}

CELEG_GGUF_AVX2_TARGET __m128i expand16(uint16_t bits) {
    const __m128i lo = expand8(static_cast<uint8_t>(bits & 0xFFu));
    const __m128i hi = expand8(static_cast<uint8_t>(bits >> 8));
    return _mm_unpacklo_epi64(lo, hi);
}

CELEG_GGUF_AVX2_TARGET int dot_i8_u4(const int8_t* activation,
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

}

CELEG_GGUF_AVX2_TARGET
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
        for (int i = 0; i < 256; i += 8) {
            const __m256 values = _mm256_mul_ps(
                _mm256_loadu_ps(source + i), inverse);
            const __m256i integers = _mm256_cvtps_epi32(values);
            const __m128i min_value = _mm_set1_epi32(-127);
            const __m128i max_value = _mm_set1_epi32(127);
            __m128i low = _mm256_castsi256_si128(integers);
            __m128i high = _mm256_extracti128_si256(integers, 1);
            low = _mm_max_epi32(min_value, _mm_min_epi32(max_value, low));
            high = _mm_max_epi32(min_value, _mm_min_epi32(max_value, high));
            const __m128i packed16 = _mm_packs_epi32(low, high);
            const __m128i packed8 = _mm_packs_epi16(packed16, packed16);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(block.qs.data() + i), packed8);
        }
        for (int group = 0; group < 16; ++group) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += block.qs[group * 16 + i];
            block.bsums[group] = static_cast<int16_t>(sum);
        }
    }
}

CELEG_GGUF_AVX2_TARGET
void cpu_gguf_dot4_avx2(const std::byte* packed_row, GgmlType type,
                        const CpuQ8KBlock* activation, size_t cols,
                        float* output4) {
    if (type == GgmlType::Q6_K) {
        const auto* weights = reinterpret_cast<const BlockQ6K*>(packed_row);
        float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const size_t blocks = cols / 256;
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i m12 = _mm256_set1_epi8(12);
        const __m256i m48 = _mm256_set1_epi8(48);
        const __m256i mhi = _mm256_set1_epi8(static_cast<char>(0xc0));
        const __m256i m15 = _mm256_set1_epi8(15);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ6K& weight = weights[b];
            int block_totals[4] = {0, 0, 0, 0};
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql = weight.ql + half * 64;
                const uint8_t* qh = weight.qh + half * 32;
                const __m256i low = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql));
                const __m256i high = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql + 32));
                const __m256i bits = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(qh));
                alignas(32) uint8_t decoded[4][32];
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[0]),
                    _mm256_or_si256(_mm256_and_si256(low, m15),
                        _mm256_slli_epi16(_mm256_and_si256(bits, m3), 4)));
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[1]),
                    _mm256_or_si256(_mm256_and_si256(high, m15),
                        _mm256_slli_epi16(_mm256_and_si256(bits, m12), 2)));
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[2]),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(low, 4), m15),
                        _mm256_and_si256(bits, m48)));
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[3]),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(high, 4), m15),
                        _mm256_srli_epi16(_mm256_and_si256(bits, mhi), 2)));
                for (int group = 0; group < 8; ++group) {
                    const int sub = half * 8 + group;
                    const int decoded_lane = group >> 1;
                    const uint8_t* values = decoded[decoded_lane] + (group & 1) * 16;
                    for (int lane = 0; lane < 4; ++lane) {
                        const CpuQ8KBlock& x = activation[lane * blocks + b];
                        const int dot = dot_u8_i8_16(values, x.qs.data() + sub * 16) -
                            32 * x.bsums[sub];
                        block_totals[lane] += static_cast<int>(weight.scales[sub]) * dot;
                    }
                }
            }
            const float d = fp16_to_float(weight.d);
            for (int lane = 0; lane < 4; ++lane) {
                totals[lane] += d * activation[lane * blocks + b].d *
                    static_cast<float>(block_totals[lane]);
            }
        }
        for (int lane = 0; lane < 4; ++lane) output4[lane] = totals[lane];
        return;
    }
    if (type != GgmlType::Q4_K) {
        for (int lane = 0; lane < 4; ++lane) {
            output4[lane] = cpu_gguf_dot_avx2(packed_row, type,
                activation + lane * (cols / 256), cols);
        }
        return;
    }
    const auto* weights = reinterpret_cast<const BlockQ4K*>(packed_row);
    float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t blocks = cols / 256;
    const __m256i mask = _mm256_set1_epi8(15);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (size_t b = 0; b < blocks; ++b) {
        const BlockQ4K& weight = weights[b];
        const float d = fp16_to_float(weight.d);
        const float dmin = fp16_to_float(weight.dmin);
        __m256i scaled[4] = {
            _mm256_setzero_si256(), _mm256_setzero_si256(),
            _mm256_setzero_si256(), _mm256_setzero_si256()};
        int minimum_totals[4] = {0, 0, 0, 0};
        for (int sub = 0; sub < 8; ++sub) {
            uint8_t scale = 0, minimum = 0;
            scale_min(weight, sub, scale, minimum);
            const uint8_t* packed = weight.qs + (sub >> 1) * 32;
            const __m256i raw = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(packed));
            const __m256i unpacked = (sub & 1) != 0
                ? _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask)
                : _mm256_and_si256(raw, mask);
            for (int lane = 0; lane < 4; ++lane) {
                const CpuQ8KBlock& x = activation[lane * blocks + b];
                const __m256i values = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(x.qs.data() + sub * 32));
                const __m256i pair = _mm256_maddubs_epi16(unpacked, values);
                const __m256i dot = _mm256_madd_epi16(pair, one16);
                scaled[lane] = _mm256_add_epi32(scaled[lane],
                    _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)));
                const int sum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                minimum_totals[lane] += static_cast<int>(minimum) * sum;
            }
        }
        for (int lane = 0; lane < 4; ++lane) {
            const float block_total = d * static_cast<float>(horizontal_sum(scaled[lane])) -
                dmin * static_cast<float>(minimum_totals[lane]);
            totals[lane] += activation[lane * blocks + b].d * block_total;
        }
    }
    for (int lane = 0; lane < 4; ++lane) output4[lane] = totals[lane];
}

CELEG_GGUF_AVX2_TARGET
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
            int scaled_total = 0;
            int minimum_total = 0;
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t scale = 0, minimum = 0;
                scale_min(weight, sub, scale, minimum);
                const int dot = dot_u8_i8_32(
                    weight.qs + (sub >> 1) * 32,
                    x.qs.data() + sub * 32, (sub & 1) != 0);
                const int sum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                scaled_total += static_cast<int>(scale) * dot;
                minimum_total += static_cast<int>(minimum) * sum;
            }
            total += x.d * (d * static_cast<float>(scaled_total) -
                            dmin * static_cast<float>(minimum_total));
        }
        return total;
    }
    if (type == GgmlType::Q6_K) {
        const auto* weights = reinterpret_cast<const BlockQ6K*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const BlockQ6K& weight = weights[b];
            const CpuQ8KBlock& x = activation[b];
            const __m256i m3 = _mm256_set1_epi8(3);
            const __m256i m12 = _mm256_set1_epi8(12);
            const __m256i m48 = _mm256_set1_epi8(48);
            const __m256i mhi = _mm256_set1_epi8(static_cast<char>(0xc0));
            const __m256i m15 = _mm256_set1_epi8(15);
            int block_total = 0;
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql = weight.ql + half * 64;
                const uint8_t* qh = weight.qh + half * 32;
                const int8_t* q8 = x.qs.data() + half * 128;
                const __m256i low = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql));
                const __m256i high = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql + 32));
                const __m256i bits = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(qh));
                const __m256i q0 = _mm256_or_si256(
                    _mm256_and_si256(low, m15),
                    _mm256_slli_epi16(_mm256_and_si256(bits, m3), 4));
                const __m256i q1 = _mm256_or_si256(
                    _mm256_and_si256(high, m15),
                    _mm256_slli_epi16(_mm256_and_si256(bits, m12), 2));
                const __m256i q2 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(low, 4), m15),
                    _mm256_and_si256(bits, m48));
                const __m256i q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(high, 4), m15),
                    _mm256_srli_epi16(_mm256_and_si256(bits, mhi), 2));
                alignas(32) uint8_t decoded[4][32];
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[0]), q0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[1]), q1);
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[2]), q2);
                _mm256_store_si256(reinterpret_cast<__m256i*>(decoded[3]), q3);
                for (int group = 0; group < 8; ++group) {
                    const int sub = half * 8 + group;
                    const int lane = group >> 1;
                    const uint8_t* values = decoded[lane] + (group & 1) * 16;
                    const int dot = dot_u8_i8_16(values, q8 + group * 16) -
                        32 * (x.bsums[sub]);
                    block_total += static_cast<int>(weight.scales[sub]) * dot;
                }
            }
            total += fp16_to_float(weight.d) * x.d *
                     static_cast<float>(block_total);
        }
        return total;
    }
    if (type == GgmlType::Q4_0) {
        const auto* weights = reinterpret_cast<const BlockQ4_0*>(packed_row);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ4_0& weight = weights[b * 8 + static_cast<size_t>(sub)];
                const int dot =
                    dot_u8_i8_16_nib(weight.qs, x.qs.data() + sub * 32, false) +
                    dot_u8_i8_16_nib(weight.qs, x.qs.data() + sub * 32 + 16, true);
                const int bsum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += fp16_to_float(weight.d) *
                              static_cast<float>(dot - 8 * bsum);
            }
            total += x.d * block_total;
        }
        return total;
    }
    if (type == GgmlType::Q5_0) {
        const auto* weights = reinterpret_cast<const BlockQ5_0*>(packed_row);
        const __m128i lo_mask = _mm_set1_epi8(15);
        for (size_t b = 0; b < blocks; ++b) {
            const CpuQ8KBlock& x = activation[b];
            float block_total = 0.0f;
            for (int sub = 0; sub < 8; ++sub) {
                const BlockQ5_0& weight = weights[b * 8 + static_cast<size_t>(sub)];
                uint32_t qh;
                std::memcpy(&qh, weight.qh, sizeof(qh));
                const __m128i hi_low = expand16(static_cast<uint16_t>(qh & 0xFFFFu));
                const __m128i hi_high = expand16(static_cast<uint16_t>(qh >> 16));
                const __m128i qs128 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(weight.qs));
                const __m128i low4 = _mm_and_si128(qs128, lo_mask);
                const __m128i high4 = _mm_and_si128(
                    _mm_srli_epi16(qs128, 4), lo_mask);
                const __m128i x_low = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x.qs.data() + sub * 32));
                const __m128i x_high = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(x.qs.data() + sub * 32 + 16));
                const __m128i vlow = _mm_or_si128(
                    low4, _mm_slli_epi16(hi_low, 4));
                const __m128i vhigh = _mm_or_si128(
                    high4, _mm_slli_epi16(hi_high, 4));
                const int dot = dot_u8_i8_16(
                                   reinterpret_cast<const uint8_t*>(&vlow),
                                   reinterpret_cast<const int8_t*>(&x_low)) +
                               dot_u8_i8_16(
                                   reinterpret_cast<const uint8_t*>(&vhigh),
                                   reinterpret_cast<const int8_t*>(&x_high));
                const int bsum = x.bsums[sub * 2] + x.bsums[sub * 2 + 1];
                block_total += fp16_to_float(weight.d) *
                              static_cast<float>(dot - 16 * bsum);
            }
            total += x.d * block_total;
        }
        return total;
    }
    return cpu_gguf_dot_scalar(packed_row, type, activation, cols);
}

#undef CELEG_GGUF_AVX2_TARGET

}

#endif
