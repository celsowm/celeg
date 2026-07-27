#include "lfm/backend/cpu/kernels.hpp"
#include "lfm/model/weights/quantization.hpp"
#include "lfm/backend/cpu/isa.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace lfm {
namespace {

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
static const bool g_has_avx2_fma = []() {
    auto caps = detect_cpu_capabilities();
    return caps.avx2 && caps.fma;
}();
#endif

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

        // Decode 16 elements (8 bytes) at a time using AVX2 SIMD
        for (; col + 16 <= group_end; col += 16) {
            __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed_row + (col >> 1)));

            __m128i low_nibbles = _mm_and_si128(raw, mask_0f);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

            __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);

            // Convert 4-bit two's complement to signed 8-bit integers:
            // interleaved holds values v in [0, 15] for each nibble.
            // XORing with 8 maps [0, 7] to [8, 15] and [8, 15] to [0, 7], effectively flipping the 4th bit.
            // Subtracting 8 then shifts them to the correct signed range [-8, 7], achieving fast branchless sign-extension.
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

        // 8-element fallback
        for (; col + 8 <= group_end; col += 8) {
            int32_t raw_val;
            std::memcpy(&raw_val, packed_row + (col >> 1), 4);
            __m128i raw = _mm_cvtsi32_si128(raw_val);

            __m128i low_nibbles = _mm_and_si128(raw, mask_0f);
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);

            __m128i interleaved = _mm_unpacklo_epi8(low_nibbles, high_nibbles);

            // Convert 4-bit two's complement to signed 8-bit integers (XOR and subtract for branchless sign-extension)
            __m128i shifted = _mm_xor_si128(interleaved, val_8);
            __m128i signed_bytes = _mm_sub_epi8(shifted, val_8);

            __m256i ints = _mm256_cvtepi8_epi32(signed_bytes);
            __m256 wf = _mm256_cvtepi32_ps(ints);
            __m256 xv = _mm256_loadu_ps(activation + col);

            group_total = _mm256_fmadd_ps(wf, xv, group_total);
        }

        total = _mm256_fmadd_ps(group_total, scale, total);

        // Scalar fallback for any remaining elements in this group
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

    // Outer scalar fallback for safety if cols exceeds groups_per_row * group_size
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
static int32_t vnni_dot_32(const uint8_t* packed,
                           const int8_t* activation) {
    const __m128i bytes = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(packed));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(bytes, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    __m256i weights = _mm256_castsi128_si256(_mm_unpacklo_epi8(low, high));
    weights = _mm256_inserti128_si256(
        weights, _mm_unpackhi_epi8(low, high), 1);
    // Toggle the sign bit of the 4-bit two's-complement value. The resulting
    // unsigned byte equals signed_q4 + 8, matching VPDPBUSD's U8 x S8 form.
    weights = _mm256_xor_si256(weights, _mm256_set1_epi8(8));
    const __m256i x = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(activation));
    __m256i acc = _mm256_setzero_si256();
    acc = _mm256_dpbusd_epi32(acc, weights, x);
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
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
                         size_t cols,
                         size_t group_size,
                         size_t groups_per_row) {
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
            // This tail is only relevant for non-standard dimensions.
            dot += (decode_q4(packed_row, col) + 8) *
                   static_cast<int32_t>(activation_q8[col]);
        }
        dot -= 8 * activation_sums[group];
        result += static_cast<float>(dot) *
                  bf16_bits_to_float(weight_scales_bf16[group]) *
                  activation_scales[group];
    }
    return result;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
float q4_q8_dot_avx512_vnni(const uint8_t* packed_row,
                            const uint16_t* weight_scales_bf16,
                            const int8_t* activation_q8,
                            const float* activation_scales,
                            const int32_t* activation_sums,
                            size_t cols,
                            size_t group_size,
                            size_t groups_per_row) {
    // AVX-512 VNNI with VL exposes the same 256-bit VPDPBUSD form. A 32-value
    // tile exactly matches the default Q4 group and avoids masked tails.
    return q4_q8_dot_avx_vnni(packed_row, weight_scales_bf16,
        activation_q8, activation_scales, activation_sums, cols,
        group_size, groups_per_row);
}
#endif

#if defined(__aarch64__)
float q4_dot_neon(const uint8_t* packed_row,
                  const uint16_t* scales_bf16,
                  const float* activation,
                  size_t cols,
                  size_t group_size,
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
        result += static_cast<float>(decode_q4(packed_row, col)) *
                  bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return result;
}
#endif

} // namespace

float q4_dot_scalar(const uint8_t* packed_row,
                    const uint16_t* scales_bf16,
                    const float* activation,
                    size_t cols,
                    size_t group_size,
                    size_t groups_per_row) {
    if (!packed_row || !scales_bf16 || !activation || group_size == 0 || groups_per_row == 0) {
        throw std::invalid_argument("invalid q4_dot arguments");
    }
    float sum = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
        const size_t group = col / group_size;
        sum += static_cast<float>(decode_q4(packed_row, col)) *
               bf16_bits_to_float(scales_bf16[group]) * activation[col];
    }
    return sum;
}

Q4DotFunction select_q4_dot_kernel(CpuIsa isa) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (isa == CpuIsa::Avx2 || isa == CpuIsa::AvxVnni ||
        isa == CpuIsa::Avx512Vnni || isa == CpuIsa::AmxInt8) {
        return q4_dot_avx2;
    }
#endif
#if defined(__aarch64__)
    if (isa == CpuIsa::Neon || isa == CpuIsa::DotProd || isa == CpuIsa::I8mm ||
        isa == CpuIsa::Sve2 || isa == CpuIsa::Sme2) {
        return q4_dot_neon;
    }
#endif
    return q4_dot_scalar;
}


float q4_q8_dot_scalar(const uint8_t* packed_row,
                       const uint16_t* weight_scales_bf16,
                       const int8_t* activation_q8,
                       const float* activation_scales,
                       const int32_t* activation_sums,
                       size_t cols,
                       size_t group_size,
                       size_t groups_per_row) {
    if (!packed_row || !weight_scales_bf16 || !activation_q8 ||
        !activation_scales || !activation_sums || group_size == 0 || groups_per_row == 0) {
        throw std::invalid_argument("invalid q4_q8_dot arguments");
    }
    float result = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        const size_t begin = group * group_size;
        const size_t end = std::min(cols, begin + group_size);
        int32_t dot = 0;
        for (size_t col = begin; col < end; ++col) {
            dot += decode_q4(packed_row, col) *
                   static_cast<int32_t>(activation_q8[col]);
        }
        result += static_cast<float>(dot) *
                  bf16_bits_to_float(weight_scales_bf16[group]) *
                  activation_scales[group];
    }
    return result;
}

Q4Q8DotFunction select_q4_q8_dot_kernel(CpuIsa isa) {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (isa == CpuIsa::Avx512Vnni) return q4_q8_dot_avx512_vnni;
    if (isa == CpuIsa::AvxVnni) return q4_q8_dot_avx_vnni;
#endif
    return nullptr;
}

CpuLinearEngine::CpuLinearEngine(CpuIsa isa, CpuThreadPool& pool)
    : isa_(isa), pool_(&pool), dot_(select_q4_dot_kernel(isa)),
      q8_dot_(select_q4_q8_dot_kernel(isa)), dynamic_q8_(q8_dot_ != nullptr) {
    const CpuCapabilities caps = detect_cpu_capabilities();
    if (!caps.supports(isa)) throw std::invalid_argument("requested CPU ISA is not supported by this host");
}

void CpuLinearEngine::gemv(const Q4GroupMatrix& weight, const float* input,
                           float* output, float beta) const {
    weight.validate();
    if (!input || !output) throw std::invalid_argument("null CPU GEMV buffer");
    const size_t row_bytes = weight.packed_values_per_row();
    const size_t grain = std::max<size_t>(1, weight.rows / std::max<size_t>(1, pool_->size() * 8));
    if (dynamic_q8_) {
        const Q8GroupVector activation =
            quantize_float_groupwise_q8(input, weight.cols, weight.group_size);
        pool_->parallel_for(0, weight.rows, grain, [&](size_t begin, size_t end) {
            for (size_t row = begin; row < end; ++row) {
                const float value = q8_dot_(
                    weight.values.data() + row * row_bytes,
                    weight.scales_bf16.data() + row * weight.groups_per_row,
                    activation.values.data(), activation.scales.data(), activation.sums.data(), weight.cols,
                    weight.group_size, weight.groups_per_row);
                output[row] = beta == 0.0f ? value : value + beta * output[row];
            }
        });
        return;
    }
    pool_->parallel_for(0, weight.rows, grain, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float value = dot_(
                weight.values.data() + row * row_bytes,
                weight.scales_bf16.data() + row * weight.groups_per_row,
                input, weight.cols, weight.group_size, weight.groups_per_row);
            output[row] = beta == 0.0f ? value : value + beta * output[row];
        }
    });
}

void CpuLinearEngine::gemm(const Q4GroupMatrix& weight, const float* input,
                           float* output, size_t rows, float beta) const {
    weight.validate();
    if ((!input || !output) && rows != 0) throw std::invalid_argument("null CPU GEMM buffer");
    if (rows == 0) return;
    const size_t row_bytes = weight.packed_values_per_row();
    std::vector<Q8GroupVector> quantized;
    if (dynamic_q8_) {
        quantized.reserve(rows);
        for (size_t r = 0; r < rows; ++r) {
            quantized.push_back(quantize_float_groupwise_q8(
                input + r * weight.cols, weight.cols, weight.group_size));
        }
    }
    // Flatten M x N so small request batches still saturate the persistent
    // thread pool. This is the CPU continuous-batching building block.
    const size_t jobs = rows * static_cast<size_t>(weight.rows);
    const size_t grain = std::max<size_t>(1, jobs / std::max<size_t>(1, pool_->size() * 16));
    pool_->parallel_for(0, jobs, grain, [&](size_t begin, size_t end) {
        for (size_t job = begin; job < end; ++job) {
            const size_t r = job / weight.rows;
            const size_t out = job % weight.rows;
            float value = 0.0f;
            if (dynamic_q8_) {
                const Q8GroupVector& a = quantized[r];
                value = q8_dot_(weight.values.data() + out * row_bytes,
                    weight.scales_bf16.data() + out * weight.groups_per_row,
                    a.values.data(), a.scales.data(), a.sums.data(), weight.cols,
                    weight.group_size, weight.groups_per_row);
            } else {
                value = dot_(weight.values.data() + out * row_bytes,
                    weight.scales_bf16.data() + out * weight.groups_per_row,
                    input + r * weight.cols, weight.cols,
                    weight.group_size, weight.groups_per_row);
            }
            float& destination = output[r * weight.rows + out];
            destination = beta == 0.0f ? value : value + beta * destination;
        }
    });
}

void CpuLinearEngine::embedding(const Q4GroupMatrix& table, int32_t token,
                                float* output) const {
    table.validate();
    if (token < 0 || token >= static_cast<int32_t>(table.rows) || !output) {
        throw std::invalid_argument("invalid CPU embedding token");
    }
    dequantize_q4_row(table, static_cast<size_t>(token), output);
}

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,fma")))
void cpu_rmsnorm_avx2(const float* input, const float* weight, float* output,
                      size_t width, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

__attribute__((target("avx2,fma")))
void cpu_residual_add_avx2(float* data, const float* residual, size_t count) {
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
}

__attribute__((target("avx2,fma")))
void cpu_swiglu_avx2(const float* gate_up, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}

__attribute__((target("avx2,fma")))
void cpu_qk_norm_rope_avx2(float* data, const float* norm_weight,
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
#endif

void cpu_rmsnorm(const float* input, const float* weight, float* output,
                 size_t width, float eps) {
    if (!input || !weight || !output || width == 0 || !(eps > 0.0f)) {
        throw std::invalid_argument("invalid RMSNorm arguments");
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_rmsnorm_avx2(input, weight, output, width, eps);
        return;
    }
#endif
    double sum = 0.0;
    for (size_t i = 0; i < width; ++i) sum += static_cast<double>(input[i]) * input[i];
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum / width) + eps);
    for (size_t i = 0; i < width; ++i) output[i] = input[i] * inv * weight[i];
}

void cpu_rmsnorm_inplace(float* data, const float* weight,
                         size_t width, float eps) {
    thread_local std::vector<float> temp;
    temp.resize(width);
    cpu_rmsnorm(data, weight, temp.data(), width, eps);
    std::copy(temp.begin(), temp.end(), data);
}

void cpu_residual_add(float* data, const float* residual, size_t count) {
    if (!data || !residual) throw std::invalid_argument("invalid residual arguments");
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_residual_add_avx2(data, residual, count);
        return;
    }
#endif
    for (size_t i = 0; i < count; ++i) data[i] += residual[i];
}

void cpu_swiglu(const float* gate_up, float* output, size_t count) {
    if (!gate_up || !output) throw std::invalid_argument("invalid SwiGLU arguments");
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_swiglu_avx2(gate_up, output, count);
        return;
    }
#endif
    for (size_t i = 0; i < count; ++i) {
        const float gate = gate_up[i];
        const float up = gate_up[count + i];
        output[i] = (gate / (1.0f + std::exp(-gate))) * up;
    }
}

void cpu_qk_norm_rope(float* data, const float* norm_weight,
                      int heads, int head_dim, int position,
                      float rope_theta, float eps) {
    if (!data || !norm_weight || heads <= 0 || head_dim <= 0 ||
        (head_dim % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid QK norm/RoPE arguments");
    }
    const int half = head_dim / 2;
    thread_local std::vector<float> cos_vals;
    thread_local std::vector<float> sin_vals;
    cos_vals.resize(static_cast<size_t>(half));
    sin_vals.resize(static_cast<size_t>(half));
    for (int d = 0; d < half; ++d) {
        const float frequency = std::pow(rope_theta, -2.0f * static_cast<float>(d) / head_dim);
        const float angle = static_cast<float>(position) * frequency;
        cos_vals[d] = std::cos(angle);
        sin_vals[d] = std::sin(angle);
    }
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (g_has_avx2_fma) {
        cpu_qk_norm_rope_avx2(data, norm_weight, cos_vals.data(), sin_vals.data(), heads, head_dim, eps);
        return;
    }
#endif
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
            const float* key = key_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
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
            const float* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
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
            const uint16_t* key = key_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += query[d] * bf16_bits_to_float(key[d]);
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
            const uint16_t* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kvh) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                destination[d] += probability * bf16_bits_to_float(value[d]);
            }
        }
    }
}

void cpu_conv_decode(const float* projected_bcx, const float* weight,
                     float* state, float* output, int hidden,
                     int cache_length, int position) {
    if (!projected_bcx || !weight || !state || !output || hidden <= 0 ||
        cache_length <= 0 || position < 0) {
        throw std::invalid_argument("invalid ShortConv arguments");
    }
    const int cursor = position % cache_length;
    for (int channel = 0; channel < hidden; ++channel) {
        const float b = projected_bcx[channel];
        const float c = projected_bcx[hidden + channel];
        const float x = projected_bcx[2 * hidden + channel];
        state[static_cast<size_t>(cursor) * hidden + channel] = b * x;
        float conv = 0.0f;
        const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
        for (int tap = 0; tap < cache_length; ++tap) {
            const int slot = (cursor + 1 + tap) % cache_length;
            conv += state[static_cast<size_t>(slot) * hidden + channel] *
                    weight[weight_offset + tap];
        }
        output[channel] = c * conv;
    }
}

} // namespace lfm
