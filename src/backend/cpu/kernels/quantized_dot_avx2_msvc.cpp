#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

#include "quantized_dot_avx2_msvc.hpp"

#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>

namespace lfm::detail {
namespace {

inline int decode_q4(const uint8_t* packed, size_t col) {
    const uint8_t byte = packed[col >> 1];
    const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

static int32_t q4_q8_dot_32_avx2(const uint8_t* packed, const int8_t* activation) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i weights = _mm256_castsi128_si256(_mm_unpacklo_epi8(low, high));
    weights = _mm256_inserti128_si256(weights, _mm_unpackhi_epi8(low, high), 1);
    weights = _mm256_xor_si256(weights, _mm256_set1_epi8(8));
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation));
    const __m256i pair_sums = _mm256_maddubs_epi16(weights, x);
    const __m256i dot32 = _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(dot32), _mm256_extracti128_si256(dot32, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

static void q4_q8_dot_32x4_avx2(const uint8_t* packed, const int8_t* activation0,
                                 const int8_t* activation1, const int8_t* activation2,
                                 const int8_t* activation3, int32_t* output4) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i weights = _mm256_castsi128_si256(_mm_unpacklo_epi8(low, high));
    weights = _mm256_inserti128_si256(weights, _mm_unpackhi_epi8(low, high), 1);
    weights = _mm256_xor_si256(weights, _mm256_set1_epi8(8));
    const __m256i ones = _mm256_set1_epi16(1);
    const auto dot = [&](const int8_t* activation) {
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation));
        const __m256i pairs = _mm256_maddubs_epi16(weights, x);
        const __m256i values = _mm256_madd_epi16(pairs, ones);
        __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(values),
                                    _mm256_extracti128_si256(values, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    };
    output4[0] = dot(activation0);
    output4[1] = dot(activation1);
    output4[2] = dot(activation2);
    output4[3] = dot(activation3);
}

} // namespace

float q4_dot_avx2_msvc(const uint8_t* packed_row, const uint16_t* scales_bf16,
                       const float* activation, size_t cols, size_t group_size,
                       size_t groups_per_row) {
    __m256 total = _mm256_setzero_ps();
    size_t col = 0;
    const __m128i mask_0f = _mm_set1_epi8(0x0f);
    const __m128i val_8 = _mm_set1_epi8(8);
    float scalar_tail = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const float scale_val = lfm::bf16_bits_to_float(scales_bf16[group]);
        const __m256 scale = _mm256_set1_ps(scale_val);
        const size_t group_end = std::min(cols, (group + 1) * group_size);
        __m256 group_total = _mm256_setzero_ps();
        for (; col + 16 <= group_end; col += 16) {
            __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_row + (col >> 1)));
            __m128i low_nibbles = _mm_and_si128(raw, mask_0f);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
            __m128i shifted = _mm_xor_si128(interleaved, val_8);
            __m128i signed_bytes = _mm_sub_epi8(shifted, val_8);
            __m256i ints_lo = _mm256_cvtepi8_epi32(signed_bytes);
            __m256i ints_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(signed_bytes, 8));
            group_total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ints_lo), _mm256_loadu_ps(activation + col), group_total);
            group_total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ints_hi), _mm256_loadu_ps(activation + col + 8), group_total);
        }
        for (; col + 8 <= group_end; col += 8) {
            int32_t raw_val;
            std::memcpy(&raw_val, packed_row + (col >> 1), 4);
            __m128i raw = _mm_cvtsi32_si128(raw_val);
            __m128i low_nibbles = _mm_and_si128(raw, mask_0f);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
            __m128i shifted = _mm_xor_si128(interleaved, val_8);
            __m128i signed_bytes = _mm_sub_epi8(shifted, val_8);
            group_total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(signed_bytes)), _mm256_loadu_ps(activation + col), group_total);
        }
        total = _mm256_fmadd_ps(group_total, scale, total);
        for (; col < group_end; ++col) scalar_tail += static_cast<float>(decode_q4(packed_row, col)) * scale_val * activation[col];
    }
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum) + scalar_tail;
    for (; col < cols; ++col) {
        const size_t group = col / group_size;
        result += static_cast<float>(decode_q4(packed_row, col)) * lfm::bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return result;
}

float q4_q8_dot_avx2_msvc(const uint8_t* packed_row, const uint16_t* weight_scales_bf16,
                          const int8_t* activation_q8, const float* activation_scales,
                          const int32_t* activation_sums, size_t cols, size_t group_size,
                          size_t groups_per_row) {
    float result = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        const size_t simd_end = begin + ((end - begin) / 32) * 32;
        int32_t dot = 0;
        for (size_t col = begin; col < simd_end; col += 32) dot += q4_q8_dot_32_avx2(packed_row + (col >> 1), activation_q8 + col);
        int32_t scalar_activation_sum = 0;
        for (size_t col = simd_end; col < end; ++col) {
            dot += decode_q4(packed_row, col) * static_cast<int32_t>(activation_q8[col]);
            scalar_activation_sum += activation_q8[col];
        }
        if (simd_end != begin) dot -= 8 * (activation_sums[group] - scalar_activation_sum);
        result += static_cast<float>(dot) * lfm::bf16_bits_to_float(weight_scales_bf16[group]) * activation_scales[group];
    }
    return result;
}

void q4_q8_dot4_avx2_msvc(const uint8_t* packed_row,
                           const uint16_t* weight_scales_bf16,
                           const Q8GroupVector& activation0,
                           const Q8GroupVector& activation1,
                           const Q8GroupVector& activation2,
                           const Q8GroupVector& activation3,
                           size_t cols, size_t group_size, size_t groups_per_row,
                           float* output4) {
    if (!packed_row || !weight_scales_bf16 || !output4 || group_size != 32 ||
        cols % 32 != 0 || activation0.elements != cols || activation1.elements != cols ||
        activation2.elements != cols || activation3.elements != cols) {
        throw std::invalid_argument("invalid AVX2 Q4xQ8x4 arguments");
    }
    const Q8GroupVector* activations[4] = {&activation0, &activation1, &activation2, &activation3};
    for (size_t row = 0; row < 4; ++row) output4[row] = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        int32_t dots[4]{};
        q4_q8_dot_32x4_avx2(packed_row + group * 16,
                             activation0.values.data() + group * 32,
                             activation1.values.data() + group * 32,
                             activation2.values.data() + group * 32,
                             activation3.values.data() + group * 32, dots);
        const float weight_scale = lfm::bf16_bits_to_float(weight_scales_bf16[group]);
        for (size_t row = 0; row < 4; ++row) {
            output4[row] += static_cast<float>(dots[row] - 8 * activations[row]->sums[group]) *
                weight_scale * activations[row]->scales[group];
        }
    }
}

} // namespace lfm::detail

#endif
