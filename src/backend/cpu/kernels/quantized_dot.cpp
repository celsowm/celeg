#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define LFM_CPU_X86 1
#include <immintrin.h>
#else
#define LFM_CPU_X86 0
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(_MSC_VER) && LFM_CPU_X86
#include "quantized_dot_avx2_msvc.hpp"
#endif

namespace lfm {
namespace {

inline int decode_q4(const uint8_t* packed, size_t col) {
    const uint8_t byte = packed[col >> 1];
    const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma")))
float q4_dot_avx2(const uint8_t* packed_row,
                  const uint16_t* scales_bf16,
                  const float* activation,
                  size_t cols,
                  size_t group_size,
                  size_t groups_per_row) {
    __m256 total = _mm256_setzero_ps();
    size_t col = 0;
    const __m128i mask_0f = _mm_set1_epi8(0x0f);
    const __m128i val_8 = _mm_set1_epi8(8);
    float scalar_tail = 0.0f;

    for (size_t group = 0; group < groups_per_row; ++group) {
        const float scale_val = bf16_bits_to_float(scales_bf16[group]);
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
            __m256 wf_lo = _mm256_cvtepi32_ps(ints_lo);
            __m256 wf_hi = _mm256_cvtepi32_ps(ints_hi);
            __m256 xv_lo = _mm256_loadu_ps(activation + col);
            __m256 xv_hi = _mm256_loadu_ps(activation + col + 8);
            group_total = _mm256_fmadd_ps(wf_lo, xv_lo, group_total);
            group_total = _mm256_fmadd_ps(wf_hi, xv_hi, group_total);
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
            __m256i ints = _mm256_cvtepi8_epi32(signed_bytes);
            __m256 wf = _mm256_cvtepi32_ps(ints);
            __m256 xv = _mm256_loadu_ps(activation + col);
            group_total = _mm256_fmadd_ps(wf, xv, group_total);
        }

        total = _mm256_fmadd_ps(group_total, scale, total);
        for (; col < group_end; ++col) {
            scalar_tail += static_cast<float>(decode_q4(packed_row, col)) * scale_val * activation[col];
        }
    }

    __m128 lo = _mm256_castps256_ps128(total);
    __m128 hi = _mm256_extractf128_ps(total, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum) + scalar_tail;
    for (; col < cols; ++col) {
        const size_t group = col / group_size;
        result += static_cast<float>(decode_q4(packed_row, col)) *
                  bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return result;
}
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,avxvnni")))
static int32_t vnni_dot_32(const uint8_t* packed, const int8_t* activation) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i weights = _mm256_castsi128_si256(_mm_unpacklo_epi8(low, high));
    weights = _mm256_inserti128_si256(weights, _mm_unpackhi_epi8(low, high), 1);
    weights = _mm256_xor_si256(weights, _mm256_set1_epi8(8));
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(activation));
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, weights, x);
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

__attribute__((target("avx2,avxvnni")))
float q4_q8_dot_avx_vnni(const uint8_t* packed_row,
                         const uint16_t* weight_scales_bf16,
                         const int8_t* activation_q8,
                         const float* activation_scales,
                         const int32_t* activation_sums,
                         size_t cols, size_t group_size, size_t groups_per_row) {
    float result = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        int32_t dot = 0;
        size_t col = begin;
        for (; col + 32 <= end; col += 32) {
            dot += vnni_dot_32(packed_row + (col >> 1), activation_q8 + col);
        }
        for (; col < end; ++col) {
            dot += (decode_q4(packed_row, col) + 8) * static_cast<int32_t>(activation_q8[col]);
        }
        dot -= 8 * activation_sums[group];
        result += static_cast<float>(dot) * bf16_bits_to_float(weight_scales_bf16[group]) * activation_scales[group];
    }
    return result;
}

__attribute__((target("avx2")))
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

__attribute__((target("avx2")))
float q4_q8_dot_avx2(const uint8_t* packed_row,
                     const uint16_t* weight_scales_bf16,
                     const int8_t* activation_q8,
                     const float* activation_scales,
                     const int32_t* activation_sums,
                     size_t cols, size_t group_size, size_t groups_per_row) {
    float result = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        const size_t simd_end = begin + ((end - begin) / 32) * 32;
        int32_t dot = 0;
        for (size_t col = begin; col < simd_end; col += 32) {
            dot += q4_q8_dot_32_avx2(packed_row + (col >> 1), activation_q8 + col);
        }
        int32_t scalar_activation_sum = 0;
        for (size_t col = simd_end; col < end; ++col) {
            dot += decode_q4(packed_row, col) * static_cast<int32_t>(activation_q8[col]);
            scalar_activation_sum += activation_q8[col];
        }
        if (simd_end != begin) dot -= 8 * (activation_sums[group] - scalar_activation_sum);
        result += static_cast<float>(dot) * bf16_bits_to_float(weight_scales_bf16[group]) * activation_scales[group];
    }
    return result;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
float q4_q8_dot_avx512_vnni(const uint8_t* packed_row,
                            const uint16_t* weight_scales_bf16,
                            const int8_t* activation_q8,
                            const float* activation_scales,
                            const int32_t* activation_sums,
                            size_t cols, size_t group_size, size_t groups_per_row) {
    return q4_q8_dot_avx_vnni(packed_row, weight_scales_bf16, activation_q8,
        activation_scales, activation_sums, cols, group_size, groups_per_row);
}
#endif

#if defined(__aarch64__)
float q4_dot_neon(const uint8_t* packed_row, const uint16_t* scales_bf16,
                  const float* activation, size_t cols, size_t group_size,
                  size_t groups_per_row) {
    (void)groups_per_row;
    float32x4_t total = vdupq_n_f32(0.0f);
    size_t col = 0;
    alignas(16) float decoded[4];
    for (; col + 4 <= cols; col += 4) {
        const size_t group = col / group_size;
        const float scale = bf16_bits_to_float(scales_bf16[group]);
        for (size_t lane = 0; lane < 4; ++lane) decoded[lane] = static_cast<float>(decode_q4(packed_row, col + lane)) * scale;
        total = vfmaq_f32(total, vld1q_f32(decoded), vld1q_f32(activation + col));
    }
    float result = vaddvq_f32(total);
    for (; col < cols; ++col) {
        const size_t group = col / group_size;
        result += static_cast<float>(decode_q4(packed_row, col)) * bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return result;
}
#endif

} // namespace

float q4_dot_scalar(const uint8_t* packed_row, const uint16_t* scales_bf16,
                    const float* activation, size_t cols, size_t group_size,
                    size_t groups_per_row) {
    if (!packed_row || !scales_bf16 || !activation || group_size == 0 || groups_per_row == 0) {
        throw std::invalid_argument("invalid q4_dot arguments");
    }
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const size_t group = col / group_size;
        sum += static_cast<float>(decode_q4(packed_row, col)) * bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return sum;
}

Q4DotFunction select_q4_dot_kernel(CpuIsa isa) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (isa == CpuIsa::Avx2 || isa == CpuIsa::AvxVnni || isa == CpuIsa::Avx512Vnni || isa == CpuIsa::AmxInt8) return q4_dot_avx2;
#elif defined(_MSC_VER) && LFM_CPU_X86
    if (isa == CpuIsa::Avx2 || isa == CpuIsa::AvxVnni || isa == CpuIsa::Avx512Vnni || isa == CpuIsa::AmxInt8) return detail::q4_dot_avx2_msvc;
#endif
#if defined(__aarch64__)
    if (isa == CpuIsa::Neon || isa == CpuIsa::DotProd || isa == CpuIsa::I8mm || isa == CpuIsa::Sve2 || isa == CpuIsa::Sme2) return q4_dot_neon;
#endif
    return q4_dot_scalar;
}

float q4_q8_dot_scalar(const uint8_t* packed_row, const uint16_t* weight_scales_bf16,
                       const int8_t* activation_q8, const float* activation_scales,
                       const int32_t* activation_sums, size_t cols, size_t group_size,
                       size_t groups_per_row) {
    if (!packed_row || !weight_scales_bf16 || !activation_q8 || !activation_scales || !activation_sums || group_size == 0 || groups_per_row == 0) {
        throw std::invalid_argument("invalid q4_q8_dot arguments");
    }
    float result = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        int32_t dot = 0;
        for (size_t col = begin; col < end; ++col) dot += decode_q4(packed_row, col) * static_cast<int32_t>(activation_q8[col]);
        result += static_cast<float>(dot) * bf16_bits_to_float(weight_scales_bf16[group]) * activation_scales[group];
    }
    return result;
}

Q4Q8DotFunction select_q4_q8_dot_kernel(CpuIsa isa) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (isa == CpuIsa::Avx512Vnni) return q4_q8_dot_avx512_vnni;
    if (isa == CpuIsa::AvxVnni) return q4_q8_dot_avx_vnni;
    if (isa == CpuIsa::Avx2) return q4_q8_dot_avx2;
#elif defined(_MSC_VER) && LFM_CPU_X86
    if (isa == CpuIsa::Avx2 || isa == CpuIsa::AvxVnni || isa == CpuIsa::Avx512Vnni || isa == CpuIsa::AmxInt8) return detail::q4_q8_dot_avx2_msvc;
#endif
    return nullptr;
}

} // namespace lfm
