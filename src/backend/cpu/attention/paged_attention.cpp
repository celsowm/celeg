#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cpu/isa.hpp"

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

size_t checked_multiply(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(what);
    }
    return a * b;
}

size_t round_up(size_t value, size_t alignment) {
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("CPU KV aligned allocation overflow");
    }
    return (value + alignment - 1) / alignment * alignment;
}

void* aligned_alloc_portable(size_t alignment, size_t size) {
#if defined(_WIN32)
    return ::_aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (::posix_memalign(&ptr, alignment, size) != 0) return nullptr;
    return ptr;
#endif
}

void aligned_free_portable(void* ptr) {
#if defined(_WIN32)
    ::_aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

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
        const float new_max = std::max(state.maximum, score);
        const float old_scale = std::isfinite(state.maximum)
            ? std::exp(state.maximum - new_max) : 0.0f;
        const float new_scale = std::exp(score - new_max);
        state.denominator = state.denominator * old_scale + new_scale;

        old_scale_vec = _mm256_set1_ps(old_scale);
        new_scale_vec = _mm256_set1_ps(new_scale);

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
                accumulator[d] = accumulator[d] * old_scale + new_scale * value[d];
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
                accumulator[d] = accumulator[d] * old_scale + new_scale * bf16_bits_to_float(value[d]);
            }
        }
        state.maximum = new_max;
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
        const float new_max = std::max(state.maximum, score);
        const float old_scale = std::isfinite(state.maximum)
            ? std::exp(state.maximum - new_max) : 0.0f;
        const float new_scale = std::exp(score - new_max);
        state.denominator = state.denominator * old_scale + new_scale;
        for (int d = 0; d < head_dim; ++d) accumulator[d] *= old_scale;
        if (pool.mode() == CpuKvCacheMode::Fp32) {
            const float* value = pool.value_fp32(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) accumulator[d] += new_scale * value[d];
        } else {
            const uint16_t* value = pool.value_bf16(page, static_cast<size_t>(local)) +
                static_cast<size_t>(kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                accumulator[d] += new_scale * bf16_bits_to_float(value[d]);
            }
        }
        state.maximum = new_max;
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
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (query_position < 0) query_position = sequence_length - 1;
    if (query_position >= sequence_length) {
        throw std::invalid_argument("paged attention query position is out of range");
    }
    const int first_token = pattern.first_candidate(query_position);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float* destination = output + static_cast<size_t>(qh) * head_dim;
        std::fill(destination, destination + head_dim, 0.0f);
        PartialAttention state;
        const size_t first_page = static_cast<size_t>(first_token) / pool.page_tokens();
        for (size_t page_index = first_page; page_index < required_pages; ++page_index) {
            const int page_start = static_cast<int>(page_index * pool.page_tokens());
            const int offset = std::max(first_token - page_start, 0);
            const int tokens_in_page = std::min<int>(
                static_cast<int>(pool.page_tokens()) - offset, sequence_length - page_start - offset);
            update_online(query, qh, kvh, head_dim, scale, pool, pages[page_index],
                          offset, offset + tokens_in_page,
                          page_start, query_position, pattern, bias,
                          destination, state);
        }
        if (state.denominator == 0.0f) {
            throw std::invalid_argument("attention pattern selected no KV tokens");
        }
        const float reciprocal = 1.0f / state.denominator;
        for (int d = 0; d < head_dim; ++d) destination[d] *= reciprocal;
    }
}

void cpu_gqa_decode_paged_parallel(
    const float* q, const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages, float* output,
    int sequence_length, int q_heads, int kv_heads, int head_dim,
    CpuThreadPool& thread_pool, CpuAttentionPattern pattern,
    CpuAttentionBias bias,
    CpuPagedAttentionOptions options,
    CpuPagedAttentionStats* stats) {
    if (options.page_tile == 0) throw std::invalid_argument("CPU attention page tile must be positive");
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    const size_t tiles = (required_pages + options.page_tile - 1) / options.page_tile;
    if (!pattern.parallel_safe() || static_cast<size_t>(sequence_length) < options.parallel_threshold ||
        thread_pool.size() == 0 || tiles == 0) {
        cpu_gqa_decode_paged(q, pool, pages, output, sequence_length,
                             q_heads, kv_heads, head_dim, pattern, bias);
        if (stats) *stats = {static_cast<size_t>(q_heads), tiles, false};
        return;
    }
    if (!q || !output || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_heads % kv_heads != 0 || pages.size() < required_pages) {
        throw std::invalid_argument("invalid parallel paged GQA arguments");
    }
    const size_t task_count = static_cast<size_t>(q_heads) * tiles;
    std::vector<PartialAttention> partial(task_count);
    std::vector<float> accumulators(task_count * static_cast<size_t>(head_dim), 0.0f);
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    thread_pool.parallel_for(0, task_count, 1, [&](size_t begin, size_t end) {
        for (size_t task = begin; task < end; ++task) {
            const int qh = static_cast<int>(task / tiles);
            const size_t tile = task % tiles;
            const int kvh = qh / queries_per_kv;
            const float* query = q + static_cast<size_t>(qh) * head_dim;
            float* accumulator = accumulators.data() + task * static_cast<size_t>(head_dim);
            PartialAttention& state = partial[task];
            const size_t first_page = tile * options.page_tile;
            const size_t last_page = std::min(required_pages, first_page + options.page_tile);
            for (size_t page_index = first_page; page_index < last_page; ++page_index) {
                const int first_token = static_cast<int>(page_index * pool.page_tokens());
                const int tokens_in_page = std::min<int>(
                    static_cast<int>(pool.page_tokens()), sequence_length - first_token);
                update_online(query, qh, kvh, head_dim, scale, pool, pages[page_index],
                              0, tokens_in_page, first_token, sequence_length - 1,
                              pattern, bias, accumulator, state);
            }
        }
    });
    for (int qh = 0; qh < q_heads; ++qh) {
        float* destination = output + static_cast<size_t>(qh) * head_dim;
        std::fill(destination, destination + head_dim, 0.0f);
        PartialAttention combined;
        for (size_t tile = 0; tile < tiles; ++tile) {
            const size_t task = static_cast<size_t>(qh) * tiles + tile;
            const PartialAttention& source = partial[task];
            if (source.denominator == 0.0f) continue;
            const float new_max = std::max(combined.maximum, source.maximum);
            const float old_scale = std::isfinite(combined.maximum)
                ? std::exp(combined.maximum - new_max) : 0.0f;
            const float source_scale = std::exp(source.maximum - new_max);
            combined.denominator = combined.denominator * old_scale +
                source.denominator * source_scale;
            const float* source_output = accumulators.data() +
                task * static_cast<size_t>(head_dim);
            for (int d = 0; d < head_dim; ++d) {
                destination[d] = destination[d] * old_scale +
                    source_output[d] * source_scale;
            }
            combined.maximum = new_max;
        }
        if (combined.denominator == 0.0f) {
            throw std::invalid_argument("attention pattern selected no KV tokens");
        }
        const float reciprocal = 1.0f / combined.denominator;
        for (int d = 0; d < head_dim; ++d) destination[d] *= reciprocal;
    }
    if (stats) *stats = {task_count, tiles, true};
}

void cpu_latent_attention_decode_paged(
    const float* query_content, const float* query_rope,
    const CpuKvPagePool& pool, std::span<const CpuKvPageId> pages,
    float* output, int sequence_length, int query_heads, int latent_rank,
    int rope_head_dim, float scale, CpuAttentionPattern pattern,
    CpuAttentionBias bias, int query_position) {
    if (!query_content || !output || sequence_length <= 0 || query_heads <= 0 ||
        latent_rank <= 0 || rope_head_dim < 0 || pool.layout().latent_width !=
            static_cast<size_t>(2 * latent_rank) ||
        pool.layout().rotary_width != static_cast<size_t>(rope_head_dim)) {
        throw std::invalid_argument("invalid CPU latent attention dimensions");
    }
    const size_t required_pages = (static_cast<size_t>(sequence_length) +
        pool.page_tokens() - 1) / pool.page_tokens();
    if (pages.size() < required_pages) throw std::invalid_argument("latent page table is incomplete");
    if (rope_head_dim != 0 && !query_rope) {
        throw std::invalid_argument("latent rotary query is required");
    }
    if (query_position < 0) query_position = sequence_length - 1;
    if (query_position >= sequence_length) throw std::invalid_argument("latent query position is out of range");
    const int first_token = pattern.first_candidate(query_position);
    for (int qh = 0; qh < query_heads; ++qh) {
        float* destination = output + static_cast<size_t>(qh) * latent_rank;
        const float* content = query_content + static_cast<size_t>(qh) * latent_rank;
        const float* rotary = query_rope ? query_rope + static_cast<size_t>(qh) * rope_head_dim : nullptr;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int key_position = first_token; key_position < sequence_length; ++key_position) {
            if (!pattern.allows(query_position, key_position)) continue;
            const size_t page_index = static_cast<size_t>(key_position) / pool.page_tokens();
            const size_t offset = static_cast<size_t>(key_position) % pool.page_tokens();
            float score = 0.0f;
            if (pool.mode() == CpuKvCacheMode::Fp32) {
                const float* key = pool.latent_key_fp32(pages[page_index], offset);
                score = std::inner_product(content, content + latent_rank, key, 0.0f);
                if (rope_head_dim != 0) {
                    const float* key_rope = pool.rotary_fp32(pages[page_index], offset);
                    score += std::inner_product(rotary, rotary + rope_head_dim, key_rope, 0.0f);
                }
            } else {
                const uint16_t* key = pool.latent_key_bf16(pages[page_index], offset);
                for (int d = 0; d < latent_rank; ++d) score += content[d] * bf16_bits_to_float(key[d]);
                if (rope_head_dim != 0) {
                    const uint16_t* key_rope = pool.rotary_bf16(pages[page_index], offset);
                    for (int d = 0; d < rope_head_dim; ++d) {
                        score += rotary[d] * bf16_bits_to_float(key_rope[d]);
                    }
                }
            }
            score = score * scale + bias.score(qh, query_position, key_position);
            maximum = std::max(maximum, score);
        }
        if (!std::isfinite(maximum)) throw std::invalid_argument("latent pattern selected no tokens");
        std::fill(destination, destination + latent_rank, 0.0f);
        float denominator = 0.0f;
        for (int key_position = first_token; key_position < sequence_length; ++key_position) {
            if (!pattern.allows(query_position, key_position)) continue;
            const size_t page_index = static_cast<size_t>(key_position) / pool.page_tokens();
            const size_t offset = static_cast<size_t>(key_position) % pool.page_tokens();
            float score = 0.0f;
            if (pool.mode() == CpuKvCacheMode::Fp32) {
                const float* key = pool.latent_key_fp32(pages[page_index], offset);
                score = std::inner_product(content, content + latent_rank, key, 0.0f);
                if (rope_head_dim != 0) {
                    const float* key_rope = pool.rotary_fp32(pages[page_index], offset);
                    score += std::inner_product(rotary, rotary + rope_head_dim, key_rope, 0.0f);
                }
            } else {
                const uint16_t* key = pool.latent_key_bf16(pages[page_index], offset);
                for (int d = 0; d < latent_rank; ++d) score += content[d] * bf16_bits_to_float(key[d]);
                if (rope_head_dim != 0) {
                    const uint16_t* key_rope = pool.rotary_bf16(pages[page_index], offset);
                    for (int d = 0; d < rope_head_dim; ++d) score += rotary[d] * bf16_bits_to_float(key_rope[d]);
                }
            }
            const float probability = std::exp(
                score * scale + bias.score(qh, query_position, key_position) - maximum);
            denominator += probability;
            if (pool.mode() == CpuKvCacheMode::Fp32) {
                const float* value = pool.latent_value_fp32(pages[page_index], offset);
                for (int d = 0; d < latent_rank; ++d) destination[d] += probability * value[d];
            } else {
                const uint16_t* value = pool.latent_value_bf16(pages[page_index], offset);
                for (int d = 0; d < latent_rank; ++d) {
                    destination[d] += probability * bf16_bits_to_float(value[d]);
                }
            }
        }
        for (int d = 0; d < latent_rank; ++d) destination[d] /= denominator;
    }
}


}

