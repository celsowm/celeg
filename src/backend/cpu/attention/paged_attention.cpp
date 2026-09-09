#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cpu/isa.hpp"
#include "celeg/attention/online_semantics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#define CELEG_CPU_HAS_AVX2_KERNEL 1
#define CELEG_CPU_AVX2_TARGET __attribute__((target("avx2,fma")))
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#pragma GCC pop_options
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define CELEG_CPU_HAS_AVX2_KERNEL 1
#define CELEG_CPU_AVX2_TARGET
#include <immintrin.h>
#else
#define CELEG_CPU_HAS_AVX2_KERNEL 0
#define CELEG_CPU_AVX2_TARGET
#endif

namespace celeg {

namespace {

#if CELEG_CPU_HAS_AVX2_KERNEL
static const bool g_has_avx2_fma = []() {
    auto caps = detect_cpu_capabilities();
    return caps.avx2 && caps.fma;
}();
#endif

struct PartialAttention {
    float maximum = -std::numeric_limits<float>::infinity();
    float denominator = 0.0f;
};

#if CELEG_CPU_HAS_AVX2_KERNEL
CELEG_CPU_AVX2_TARGET
void update_online_avx2(const float* query, int kv_head, int head_dim, float scale,
                        const CpuKvPagePool& pool, CpuKvPageId page,
                        int token_begin, int token_end, float* accumulator,
                        PartialAttention& state) {
    __m256 old_scale_vec = _mm256_setzero_ps();
    __m256 new_scale_vec = _mm256_setzero_ps();

    for (int local = token_begin; local < token_end; ++local) {
        float dot = 0.0f;
        if (pool.mode() == CpuKvCacheMode::Fp32) {
            const float* key = pool.key_fp32(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            __m256 dot_vec = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 q_val = _mm256_loadu_ps(query + d);
                __m256 k_val = _mm256_loadu_ps(key + d);
                dot_vec = _mm256_fmadd_ps(q_val, k_val, dot_vec);
            }
            __m128 lo = _mm256_castps256_ps128(dot_vec);
            __m128 hi = _mm256_extractf128_ps(dot_vec, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            dot = _mm_cvtss_f32(sum);
            for (; d < head_dim; ++d) {
                dot += query[d] * key[d];
            }
        } else {
            const uint16_t* key = pool.key_bf16(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            __m256 dot_vec = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 q_val = _mm256_loadu_ps(query + d);
                __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key + d));
                __m256 k_val = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
                dot_vec = _mm256_fmadd_ps(q_val, k_val, dot_vec);
            }
            __m128 lo = _mm256_castps256_ps128(dot_vec);
            __m128 hi = _mm256_extractf128_ps(dot_vec, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            dot = _mm_cvtss_f32(sum);
            for (; d < head_dim; ++d) {
                dot += query[d] * bf16_bits_to_float(key[d]);
            }
        }

        const float score = dot * scale;
        const auto transition = attention_semantics::online_transition(
            state.maximum, state.denominator, score);
        state.denominator = transition.denominator;

        old_scale_vec = _mm256_set1_ps(transition.previous_scale);
        new_scale_vec = _mm256_set1_ps(transition.current_scale);

        int d = 0;
        if (pool.mode() == CpuKvCacheMode::Fp32) {
            const float* value = pool.value_fp32(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 acc_val = _mm256_loadu_ps(accumulator + d);
                __m256 val_val = _mm256_loadu_ps(value + d);
                acc_val = _mm256_mul_ps(acc_val, old_scale_vec);
                acc_val = _mm256_fmadd_ps(val_val, new_scale_vec, acc_val);
                _mm256_storeu_ps(accumulator + d, acc_val);
            }
            for (; d < head_dim; ++d) {
                accumulator[d] = attention_semantics::online_accumulate(
                    accumulator[d], value[d], transition);
            }
        } else {
            const uint16_t* value = pool.value_bf16(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 acc_val = _mm256_loadu_ps(accumulator + d);
                __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(value + d));
                __m256 val_val = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
                acc_val = _mm256_mul_ps(acc_val, old_scale_vec);
                acc_val = _mm256_fmadd_ps(val_val, new_scale_vec, acc_val);
                _mm256_storeu_ps(accumulator + d, acc_val);
            }
            for (; d < head_dim; ++d) {
                accumulator[d] = attention_semantics::online_accumulate(
                    accumulator[d], bf16_bits_to_float(value[d]), transition);
            }
        }
        state.maximum = transition.maximum;
    }
}
#endif

void update_online(const float* query, int query_head, int kv_head, int head_dim, float scale,
                   const CpuKvPagePool& pool, CpuKvPageId page,
                   int token_begin, int token_end, int page_token_base,
                   int query_position, const CpuAttentionPattern& pattern,
                   const CpuAttentionBias& bias, float* accumulator,
                   PartialAttention& state) {
#if CELEG_CPU_HAS_AVX2_KERNEL
    if (g_has_avx2_fma && bias.empty() && pattern.parallel_safe()) {
        update_online_avx2(query, kv_head, head_dim, scale, pool, page,
                           token_begin, token_end, accumulator, state);
        return;
    }
#endif

    for (int local = token_begin; local < token_end; ++local) {
        if (!pattern.allows(query_position, page_token_base + local)) continue;
        float dot = 0.0f;
        if (pool.mode() == CpuKvCacheMode::Fp32) {
            const float* key = pool.key_fp32(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) dot += query[d] * key[d];
        } else {
            const uint16_t* key = pool.key_bf16(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                dot += query[d] * bf16_bits_to_float(key[d]);
            }
        }
        const float score = dot * scale +
            bias.score(query_head, query_position, page_token_base + local);
        const auto transition = attention_semantics::online_transition(
            state.maximum, state.denominator, score);
        state.denominator = transition.denominator;
        if (pool.mode() == CpuKvCacheMode::Fp32) {
            const float* value = pool.value_fp32(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                accumulator[d] = attention_semantics::online_accumulate(
                    accumulator[d], value[d], transition);
            }
        } else {
            const uint16_t* value = pool.value_bf16(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                accumulator[d] = attention_semantics::online_accumulate(
                    accumulator[d], bf16_bits_to_float(value[d]), transition);
            }
        }
        state.maximum = transition.maximum;
    }
}
}


void cpu_gqa_decode_paged(const float* q,
                          const CpuKvPagePool& pool,
                          std::span<const CpuKvPageId> pages,
                          float* output,
                          int sequence_length,
                          int q_heads,
                          int kv_heads,
                          int head_dim,
                          CpuAttentionPattern pattern,
                          CpuAttentionBias bias,
                          int query_position) {
    if (!q || !output) throw std::invalid_argument("paged GQA pointers are required");
    if (sequence_length <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_heads % kv_heads != 0) {
        throw std::invalid_argument("invalid paged GQA dimensions");
    }
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    if (pages.size() < required_pages) throw std::invalid_argument("paged GQA page table is incomplete");
    if (pool.key_width() != static_cast<size_t>(kv_heads * head_dim) ||
        pool.value_width() != static_cast<size_t>(kv_heads * head_dim)) {
        throw std::invalid_argument("paged GQA KV width mismatch");
    }
    // ... remainder unchanged ...
}
