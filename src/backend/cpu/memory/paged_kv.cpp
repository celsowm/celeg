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

// The online-softmax attention kernel below is AVX2/FMA.  GCC and Clang need
// the target attribute to emit those instructions from a generically compiled
// translation unit; MSVC accepts the intrinsics directly, so it only needs the
// runtime capability check.  Without this the whole attention inner loop fell
// back to scalar code on MSVC builds.
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
    if (g_has_avx2_fma && bias.empty() &&
        pattern.kind == CpuAttentionPatternKind::FullCausal) {
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

void CpuStatePageLayout::validate() const {
    if (token_elements() == 0) {
        throw std::invalid_argument("CPU state page layout must contain state");
    }
}

struct CpuKvPagePool::Page {
    size_t references = 0;
    void* storage = nullptr;
    size_t storage_bytes = 0;
    int numa_node = -1;
    bool numa_bound = false;
    bool numa_binding_failed = false;

    ~Page() { aligned_free_portable(storage); }
};

CpuKvPagePool::CpuKvPagePool(CpuKvCacheMode mode,
                             size_t page_tokens,
                             CpuStatePageLayout layout)
    : mode_(mode), page_tokens_(page_tokens), layout_(layout) {
    if (page_tokens_ == 0) {
        throw std::invalid_argument("CPU state page token count must be positive");
    }
    layout_.validate();
    const size_t elements = checked_multiply(
        page_tokens_, layout_.token_elements(), "CPU state page dimensions overflow");
    page_bytes_ = checked_multiply(elements,
        mode_ == CpuKvCacheMode::Fp32 ? sizeof(float) : sizeof(uint16_t),
        "CPU KV page byte size overflow");
}

CpuKvPagePool::~CpuKvPagePool() = default;

CpuKvPageId CpuKvPagePool::allocate(int requested_node) {
    std::lock_guard lock(mutex_);
    CpuKvPageId id = kInvalidCpuKvPage;
    if (!free_pages_.empty()) {
        auto selected = free_pages_.end();
        if (requested_node >= 0) {
            selected = std::find_if(free_pages_.begin(), free_pages_.end(),
                [&](CpuKvPageId candidate) {
                    return pages_[candidate] && pages_[candidate]->numa_node == requested_node;
                });
        }
        if (selected == free_pages_.end()) selected = std::prev(free_pages_.end());
        id = *selected;
        free_pages_.erase(selected);
    } else {
        if (pages_.size() >= static_cast<size_t>(kInvalidCpuKvPage)) {
            throw std::overflow_error("CPU KV page id space exhausted");
        }
        id = static_cast<CpuKvPageId>(pages_.size());
        pages_.push_back(std::make_unique<Page>());
    }
    Page& page = *pages_[id];
    if (page.references != 0) {
        throw std::logic_error("CPU KV free page has a non-zero reference count");
    }
    constexpr size_t alignment = 4096;
    const size_t allocation_bytes = round_up(page_bytes_, alignment);
    if (!page.storage || page.storage_bytes != allocation_bytes ||
        (requested_node >= 0 && page.numa_node != requested_node)) {
        aligned_free_portable(page.storage);
        page.storage = nullptr;
        void* memory = aligned_alloc_portable(alignment, allocation_bytes);
        if (!memory) {
            throw std::bad_alloc();
        }
        page.storage = memory;
        page.storage_bytes = allocation_bytes;
        page.numa_node = requested_node;
        page.numa_bound = requested_node >= 0 &&
            bind_memory_to_numa_node(memory, allocation_bytes, requested_node);
        page.numa_binding_failed = requested_node >= 0 && !page.numa_bound;
    }
    std::memset(page.storage, 0, page.storage_bytes);
    page.references = 1;
    return id;
}

void CpuKvPagePool::retain(CpuKvPageId page_id) {
    std::lock_guard lock(mutex_);
    Page& page = checked_page(page_id);
    if (page.references == 0) throw std::logic_error("retaining a free CPU KV page");
    if (page.references == std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("CPU KV page reference count overflow");
    }
    ++page.references;
}

void CpuKvPagePool::release(CpuKvPageId page_id) {
    std::lock_guard lock(mutex_);
    Page& page = checked_page(page_id);
    if (page.references == 0) throw std::logic_error("double release of CPU KV page");
    --page.references;
    if (page.references == 0) free_pages_.push_back(page_id);
}

size_t CpuKvPagePool::reference_count(CpuKvPageId page_id) const {
    std::lock_guard lock(mutex_);
    return checked_page(page_id).references;
}

int CpuKvPagePool::numa_node(CpuKvPageId page_id) const {
    std::lock_guard lock(mutex_);
    return checked_page(page_id).numa_node;
}

CpuKvPageId CpuKvPagePool::clone_prefix(CpuKvPageId source,
                                        size_t used_tokens,
                                        int requested_node) {
    if (used_tokens > page_tokens_) {
        throw std::out_of_range("CPU KV clone token count exceeds page size");
    }
    const int node = requested_node >= 0 ? requested_node : numa_node(source);
    const CpuKvPageId destination = allocate(node);
    try {
        const size_t element_bytes = mode_ == CpuKvCacheMode::Fp32
            ? sizeof(float) : sizeof(uint16_t);
        const size_t key_bytes = used_tokens * layout_.key_width * element_bytes;
        const size_t value_bytes = used_tokens * layout_.value_width * element_bytes;
        const Page& source_page = checked_page(source);
        Page& destination_page = checked_page(destination);
        const auto* src = static_cast<const std::byte*>(source_page.storage);
        auto* dst = static_cast<std::byte*>(destination_page.storage);
        std::memcpy(dst, src, key_bytes);
        const size_t value_base = page_tokens_ * layout_.key_width * element_bytes;
        std::memcpy(dst + value_base, src + value_base, value_bytes);
        const size_t state_base = page_tokens_ *
            (layout_.key_width + layout_.value_width) * element_bytes;
        const size_t latent_bytes = used_tokens *
            (layout_.latent_width + layout_.rotary_width) * element_bytes;
        if (latent_bytes != 0) std::memcpy(dst + state_base, src + state_base, latent_bytes);
        return destination;
    } catch (...) {
        release(destination);
        throw;
    }
}

void CpuKvPagePool::write(CpuKvPageId page_id, size_t token_offset,
                          const float* key, const float* value) {
    if (!key || !value) throw std::invalid_argument("CPU KV write pointers are required");
    if (layout_.key_width == 0 || layout_.value_width == 0) {
        throw std::logic_error("CPU state page has no ordinary KV regions");
    }
    if (token_offset >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    Page& page = checked_page(page_id);
    if (page.references == 0) throw std::logic_error("writing to a free CPU KV page");
    const size_t key_offset = token_offset * layout_.key_width;
    const size_t value_offset = page_tokens_ * layout_.key_width +
        token_offset * layout_.value_width;
    if (mode_ == CpuKvCacheMode::Fp32) {
        auto* data = static_cast<float*>(page.storage);
        std::copy(key, key + layout_.key_width, data + key_offset);
        std::copy(value, value + layout_.value_width, data + value_offset);
    } else {
        auto* data = static_cast<uint16_t*>(page.storage);
        for (size_t i = 0; i < layout_.key_width; ++i) {
            data[key_offset + i] = float_to_bf16_bits(key[i]);
        }
        for (size_t i = 0; i < layout_.value_width; ++i) {
            data[value_offset + i] = float_to_bf16_bits(value[i]);
        }
    }
}

void CpuKvPagePool::write_latent(CpuKvPageId page_id, size_t token_offset,
                                 const float* key, const float* value,
                                 const float* rotary) {
    if (!key || !value) throw std::invalid_argument("CPU latent state pointers are required");
    if (layout_.latent_width == 0 || (layout_.latent_width % 2) != 0) {
        throw std::logic_error("CPU latent state width must contain key and value regions");
    }
    if (layout_.rotary_width != 0 && !rotary) {
        throw std::invalid_argument("CPU latent rotary pointer is required");
    }
    if (token_offset >= page_tokens_) throw std::out_of_range("CPU latent token offset out of range");
    Page& page = checked_page(page_id);
    if (page.references == 0) throw std::logic_error("writing to a free CPU state page");
    const size_t width = layout_.latent_width / 2;
    const size_t state_base = page_tokens_ * (layout_.key_width + layout_.value_width);
    const size_t key_offset = state_base + token_offset * width;
    const size_t value_offset = state_base + page_tokens_ * width + token_offset * width;
    const size_t rotary_offset = state_base + page_tokens_ * layout_.latent_width +
        token_offset * layout_.rotary_width;
    if (mode_ == CpuKvCacheMode::Fp32) {
        auto* data = static_cast<float*>(page.storage);
        std::copy(key, key + width, data + key_offset);
        std::copy(value, value + width, data + value_offset);
        if (layout_.rotary_width != 0) {
            std::copy(rotary, rotary + layout_.rotary_width, data + rotary_offset);
        }
    } else {
        auto* data = static_cast<uint16_t*>(page.storage);
        for (size_t i = 0; i < width; ++i) {
            data[key_offset + i] = float_to_bf16_bits(key[i]);
            data[value_offset + i] = float_to_bf16_bits(value[i]);
        }
        for (size_t i = 0; i < layout_.rotary_width; ++i) {
            data[rotary_offset + i] = float_to_bf16_bits(rotary[i]);
        }
    }
}

const float* CpuKvPagePool::key_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const float*>(checked_page(id).storage) + token * layout_.key_width;
}
const float* CpuKvPagePool::value_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const float*>(checked_page(id).storage) +
        page_tokens_ * layout_.key_width + token * layout_.value_width;
}
const uint16_t* CpuKvPagePool::key_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) + token * layout_.key_width;
}
const uint16_t* CpuKvPagePool::value_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) +
        page_tokens_ * layout_.key_width + token * layout_.value_width;
}

const float* CpuKvPagePool::latent_key_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (layout_.latent_width == 0 || (layout_.latent_width % 2) != 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent token is out of range");
    return static_cast<const float*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width) +
        token * (layout_.latent_width / 2);
}
const float* CpuKvPagePool::latent_value_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (layout_.latent_width == 0 || (layout_.latent_width % 2) != 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent token is out of range");
    return static_cast<const float*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width + layout_.latent_width / 2) +
        token * (layout_.latent_width / 2);
}
const float* CpuKvPagePool::rotary_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (layout_.rotary_width == 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent rotary token is out of range");
    return static_cast<const float*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width + layout_.latent_width) +
        token * layout_.rotary_width;
}
const uint16_t* CpuKvPagePool::latent_key_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (layout_.latent_width == 0 || (layout_.latent_width % 2) != 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent token is out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width) +
        token * (layout_.latent_width / 2);
}
const uint16_t* CpuKvPagePool::latent_value_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (layout_.latent_width == 0 || (layout_.latent_width % 2) != 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent token is out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width + layout_.latent_width / 2) +
        token * (layout_.latent_width / 2);
}
const uint16_t* CpuKvPagePool::rotary_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (layout_.rotary_width == 0 || token >= page_tokens_)
        throw std::out_of_range("CPU latent rotary token is out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) +
        page_tokens_ * (layout_.key_width + layout_.value_width + layout_.latent_width) +
        token * layout_.rotary_width;
}

const CpuKvPagePool::Page& CpuKvPagePool::checked_page(CpuKvPageId id) const {
    if (id == kInvalidCpuKvPage || static_cast<size_t>(id) >= pages_.size() || !pages_[id]) {
        throw std::out_of_range("invalid CPU KV page id");
    }
    return *pages_[id];
}
CpuKvPagePool::Page& CpuKvPagePool::checked_page(CpuKvPageId id) {
    return const_cast<Page&>(static_cast<const CpuKvPagePool*>(this)->checked_page(id));
}

CpuKvPageStats CpuKvPagePool::stats() const {
    std::lock_guard lock(mutex_);
    CpuKvPageStats result;
    result.total_pages = pages_.size();
    result.bytes_reserved = pages_.size() * page_bytes_;
    for (const auto& page : pages_) {
        if (page && page->references != 0) {
            ++result.used_pages;
            result.retained_references += page->references;
        }
        if (page && page->numa_bound) ++result.numa_bound_pages;
        if (page && page->numa_binding_failed) ++result.numa_binding_failures;
    }
    return result;
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

CpuExternalAttentionMemory::CpuExternalAttentionMemory(
    CpuKvCacheMode mode, size_t page_tokens, size_t key_width,
    size_t value_width)
    : pool_(mode, page_tokens, CpuStatePageLayout{key_width, value_width, 0, 0}) {}

void CpuExternalAttentionMemory::append(const float* key, const float* value) {
    if (!key || !value) throw std::invalid_argument("external attention memory pointers are required");
    if (token_count_ % pool_.page_tokens() == 0) pages_.push_back(pool_.allocate());
    pool_.write(pages_.back(), token_count_ % pool_.page_tokens(), key, value);
    ++token_count_;
}

void CpuExternalAttentionMemory::clear() noexcept {
    for (CpuKvPageId page : pages_) {
        try { pool_.release(page); } catch (...) {}
    }
    pages_.clear();
    token_count_ = 0;
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

void cpu_gqa_prefill_paged(
    const float* queries, size_t query_rows, size_t query_stride,
    const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages, float* output,
    int base_sequence_length, int q_heads, int kv_heads, int head_dim,
    CpuThreadPool& thread_pool, CpuAttentionPattern pattern,
    CpuAttentionBias bias) {
    if (!queries || !output || query_rows == 0 || base_sequence_length < 0 ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0) {
        throw std::invalid_argument("invalid CPU paged prefill GQA arguments");
    }
    const size_t query_width = static_cast<size_t>(q_heads) * head_dim;
    if (query_stride < query_width) {
        throw std::invalid_argument("CPU paged prefill GQA query stride is too small");
    }
    // One parallel region covers the complete chunk. Each query writes its own
    // output row, so the only shared state is immutable KV. Patterns that can
    // read future tokens use the complete committed chunk as their horizon.
    const int committed_sequence_length = base_sequence_length +
        static_cast<int>(query_rows);
    thread_pool.parallel_for(0, query_rows, 1, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const int query_position = base_sequence_length + static_cast<int>(row);
            const int sequence_length = pattern.may_read_future(
                query_position, committed_sequence_length)
                ? committed_sequence_length : query_position + 1;
            cpu_gqa_decode_paged(queries + row * query_stride, pool, pages,
                                 output + row * query_width, sequence_length,
                                 q_heads, kv_heads, head_dim, pattern,
                                 bias, query_position);
        }
    });
}

} // namespace celeg
