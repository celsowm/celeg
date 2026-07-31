#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

#include "quantized_dot_avx2_msvc.hpp"

#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace celeg::detail {
namespace {

inline int decode_q4(const uint8_t* packed, size_t col) {
    const uint8_t byte = packed[col >> 1];
    const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

// Unpack 32 packed Q4 nibbles into 32 lanes biased by +8, so the values are
// unsigned in [0, 15] and usable as the first operand of _mm256_maddubs_epi16.
// The +8 bias is removed once per row through the activation group sums.
inline __m256i unpack_q4_biased(const uint8_t* packed) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i weights = _mm256_castsi128_si256(_mm_unpacklo_epi8(low, high));
    weights = _mm256_inserti128_si256(weights, _mm_unpackhi_epi8(low, high), 1);
    return _mm256_xor_si256(weights, _mm256_set1_epi8(8));
}

// 32 int8 products reduced to 8 int32 lanes. Deliberately *not* reduced to a
// scalar: the caller keeps the lanes live and folds them into a vector float
// accumulator, so a row pays one horizontal reduction instead of one per group.
// Biased weights are in [0, 15] and activations in [-127, 127], so the
// intermediate int16 pair sums stay well inside range.
inline __m256i dot32_epi32(__m256i biased_weights, const int8_t* activation) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation));
    const __m256i pairs = _mm256_maddubs_epi16(biased_weights, x);
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
}

inline float hsum256_ps(__m256 value) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                            _mm256_extractf128_ps(value, 1));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 0x1));
    return _mm_cvtss_f32(sum);
}

} // namespace

float f32_dot_avx2_msvc(const float* weight, const float* activation,
                         size_t cols) {
    __m256 total = _mm256_setzero_ps();
    size_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        total = _mm256_fmadd_ps(_mm256_loadu_ps(weight + col),
                                _mm256_loadu_ps(activation + col), total);
    }
    float sum = hsum256_ps(total);
    for (; col < cols; ++col) sum += weight[col] * activation[col];
    return sum;
}

float q4_dot_avx2_msvc(const uint8_t* packed_row, const uint16_t* scales_bf16,
                       const float* activation, size_t cols, size_t group_size,
                       size_t groups_per_row) {
    __m256 total = _mm256_setzero_ps();
    size_t col = 0;
    const __m128i mask_0f = _mm_set1_epi8(0x0f);
    const __m128i val_8 = _mm_set1_epi8(8);
    float scalar_tail = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const float scale_val = celeg::bf16_bits_to_float(scales_bf16[group]);
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
    float result = hsum256_ps(total) + scalar_tail;
    for (; col < cols; ++col) {
        const size_t group = col / group_size;
        result += static_cast<float>(decode_q4(packed_row, col)) * celeg::bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return result;
}

float q4_q8_dot_avx2_msvc(const uint8_t* packed_row, const uint16_t* weight_scales_bf16,
                          const int8_t* activation_q8, const float* activation_scales,
                          const int32_t* activation_sums, size_t cols, size_t group_size,
                          size_t groups_per_row) {
    __m256 accumulator = _mm256_setzero_ps();
    // Bias from the +8 weight offset, deferred to a single correction per row.
    float bias = 0.0f;
    float scalar_total = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        const size_t simd_end = begin + ((end - begin) / 32) * 32;
        const float combined = celeg::bf16_bits_to_float(weight_scales_bf16[group]) *
                               activation_scales[group];
        if (simd_end != begin) {
            __m256i group_dot = _mm256_setzero_si256();
            for (size_t col = begin; col < simd_end; col += 32) {
                group_dot = _mm256_add_epi32(group_dot,
                    dot32_epi32(unpack_q4_biased(packed_row + (col >> 1)),
                                activation_q8 + col));
            }
            accumulator = _mm256_fmadd_ps(_mm256_cvtepi32_ps(group_dot),
                                          _mm256_set1_ps(combined), accumulator);
            bias += combined * static_cast<float>(activation_sums[group]);
        }
        int32_t scalar_dot = 0;
        int32_t scalar_activation_sum = 0;
        for (size_t col = simd_end; col < end; ++col) {
            scalar_dot += decode_q4(packed_row, col) * static_cast<int32_t>(activation_q8[col]);
            scalar_activation_sum += activation_q8[col];
        }
        if (scalar_dot != 0) scalar_total += static_cast<float>(scalar_dot) * combined;
        // The SIMD span only covered part of the group, so the +8 correction
        // must exclude the activations handled by the scalar tail above.
        if (simd_end != begin && scalar_activation_sum != 0) {
            bias -= combined * static_cast<float>(scalar_activation_sum);
        }
    }
    return hsum256_ps(accumulator) - 8.0f * bias + scalar_total;
}

void q4_q8_dot4_avx2_msvc(const uint8_t* packed_row,
                           const uint16_t* weight_scales_bf16,
                           const int8_t* activation_values, size_t value_stride,
                           const float* activation_scales,
                           const int32_t* activation_sums, size_t group_stride,
                           size_t cols, size_t group_size, size_t groups_per_row,
                           float* output4) {
    if (!packed_row || !weight_scales_bf16 || !activation_values ||
        !activation_scales || !activation_sums || !output4 ||
        group_size != 32 || cols % 32 != 0) {
        throw std::invalid_argument("invalid AVX2 Q4xQ8x4 arguments");
    }
    __m256 accumulator[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                             _mm256_setzero_ps(), _mm256_setzero_ps()};
    float bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t group = 0; group < groups_per_row; ++group) {
        // One unpack of the packed nibbles feeds all four activation rows.
        const __m256i weights = unpack_q4_biased(packed_row + group * 16);
        const float weight_scale = celeg::bf16_bits_to_float(weight_scales_bf16[group]);
        for (size_t lane = 0; lane < 4; ++lane) {
            const __m256i dot = dot32_epi32(weights,
                activation_values + lane * value_stride + group * 32);
            const float combined =
                weight_scale * activation_scales[lane * group_stride + group];
            accumulator[lane] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot),
                _mm256_set1_ps(combined), accumulator[lane]);
            bias[lane] += combined *
                static_cast<float>(activation_sums[lane * group_stride + group]);
        }
    }
    for (size_t lane = 0; lane < 4; ++lane) {
        output4[lane] = hsum256_ps(accumulator[lane]) - 8.0f * bias[lane];
    }
}

} // namespace celeg::detail

#endif
