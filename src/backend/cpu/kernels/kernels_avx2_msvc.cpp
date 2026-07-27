// AVX2/FMA kernel bodies for MSVC.
//
// GCC/Clang select these at runtime via per-function
// __attribute__((target("avx2,fma"))) multiversioning, compiled alongside the
// scalar fallback in the same translation unit (kernels.cpp). MSVC has no
// per-function target attribute -- its equivalent is simply calling the
// AVX2/FMA intrinsics directly, which MSVC compiles to the corresponding
// instructions at the call site regardless of the /arch flag used for the
// rest of the file. That keeps the rest of the binary on its SSE2 baseline
// (so it still runs on CPUs without AVX2) while these specific functions,
// invoked only when detect_cpu_capabilities() confirms AVX2+FMA, use it.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

#include "kernels_avx2_msvc.hpp"

#include "lfm/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace lfm::detail {
namespace {

inline int decode_q4(const uint8_t* packed, size_t col) {
    const uint8_t byte = packed[col >> 1];
    const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

} // namespace

float q4_dot_avx2_msvc(const uint8_t* packed_row,
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
                  lfm::bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }

    return result;
}

void cpu_rmsnorm_avx2_msvc(const float* input, const float* weight, float* output,
                           size_t width, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

void cpu_residual_add_avx2_msvc(float* data, const float* residual, size_t count) {
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
}

void cpu_swiglu_avx2_msvc(const float* gate_up, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}

void cpu_qk_norm_rope_avx2_msvc(float* data, const float* norm_weight,
                                const float* cos_vals, const float* sin_vals,
                                int heads, int head_dim, float eps) {
    const int half = head_dim / 2;
    for (int head = 0; head < heads; ++head) {
        float* vector = data + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        for (int d = 0; d < head_dim; ++d) sum += static_cast<double>(vector[d]) * vector[d];
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        for (int d = 0; d < half; ++d) {
            const float a = vector[d] * inv * norm_weight[d];
            const float b = vector[d + half] * inv * norm_weight[d + half];
            vector[d] = a * cos_vals[d] - b * sin_vals[d];
            vector[d + half] = b * cos_vals[d] + a * sin_vals[d];
        }
    }
}

} // namespace lfm::detail

#endif // defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
