#include "lfm/cpu_paged_kv.hpp"
#include "lfm/cpu_numa.hpp"
#include "lfm/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfm {

namespace {
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

void update_online(const float* query, int kv_head, int head_dim, float scale,
                   const CpuKvPagePool& pool, CpuKvPageId page,
                   int token_begin, int token_end, float* accumulator,
                   PartialAttention& state) {
    for (int local = token_begin; local < token_end; ++local) {
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
        const float score = dot * scale;
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
                             size_t kv_width)
    : mode_(mode), page_tokens_(page_tokens), kv_width_(kv_width) {
    if (page_tokens_ == 0 || kv_width_ == 0) {
        throw std::invalid_argument("CPU KV page dimensions must be positive");
    }
    const size_t elements = checked_multiply(
        checked_multiply(page_tokens_, kv_width_, "CPU KV page dimensions overflow"),
        size_t{2}, "CPU KV page dimensions overflow");
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
        const size_t valid_bytes = used_tokens * kv_width_ * element_bytes;
        const Page& source_page = checked_page(source);
        Page& destination_page = checked_page(destination);
        const auto* src = static_cast<const std::byte*>(source_page.storage);
        auto* dst = static_cast<std::byte*>(destination_page.storage);
        std::memcpy(dst, src, valid_bytes);
        const size_t value_base = page_tokens_ * kv_width_ * element_bytes;
        std::memcpy(dst + value_base, src + value_base, valid_bytes);
        return destination;
    } catch (...) {
        release(destination);
        throw;
    }
}

void CpuKvPagePool::write(CpuKvPageId page_id, size_t token_offset,
                          const float* key, const float* value) {
    if (!key || !value) throw std::invalid_argument("CPU KV write pointers are required");
    if (token_offset >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    Page& page = checked_page(page_id);
    if (page.references == 0) throw std::logic_error("writing to a free CPU KV page");
    const size_t key_offset = token_offset * kv_width_;
    const size_t value_offset = page_tokens_ * kv_width_ + key_offset;
    if (mode_ == CpuKvCacheMode::Fp32) {
        auto* data = static_cast<float*>(page.storage);
        std::copy(key, key + kv_width_, data + key_offset);
        std::copy(value, value + kv_width_, data + value_offset);
    } else {
        auto* data = static_cast<uint16_t*>(page.storage);
        for (size_t i = 0; i < kv_width_; ++i) {
            data[key_offset + i] = float_to_bf16_bits(key[i]);
            data[value_offset + i] = float_to_bf16_bits(value[i]);
        }
    }
}

const float* CpuKvPagePool::key_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const float*>(checked_page(id).storage) + token * kv_width_;
}
const float* CpuKvPagePool::value_fp32(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Fp32) throw std::logic_error("CPU KV pool is not FP32");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const float*>(checked_page(id).storage) +
        page_tokens_ * kv_width_ + token * kv_width_;
}
const uint16_t* CpuKvPagePool::key_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) + token * kv_width_;
}
const uint16_t* CpuKvPagePool::value_bf16(CpuKvPageId id, size_t token) const {
    if (mode_ != CpuKvCacheMode::Bf16) throw std::logic_error("CPU KV pool is not BF16");
    if (token >= page_tokens_) throw std::out_of_range("CPU KV token offset out of range");
    return static_cast<const uint16_t*>(checked_page(id).storage) +
        page_tokens_ * kv_width_ + token * kv_width_;
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
                          int head_dim) {
    if (!q || !output) throw std::invalid_argument("paged GQA pointers are required");
    if (sequence_length <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        q_heads % kv_heads != 0) {
        throw std::invalid_argument("invalid paged GQA dimensions");
    }
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    if (pages.size() < required_pages) throw std::invalid_argument("paged GQA page table is incomplete");
    if (pool.kv_width() != static_cast<size_t>(kv_heads * head_dim)) {
        throw std::invalid_argument("paged GQA KV width mismatch");
    }
    const int queries_per_kv = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / queries_per_kv;
        const float* query = q + static_cast<size_t>(qh) * head_dim;
        float* destination = output + static_cast<size_t>(qh) * head_dim;
        std::fill(destination, destination + head_dim, 0.0f);
        PartialAttention state;
        int processed = 0;
        for (size_t page_index = 0; page_index < required_pages; ++page_index) {
            const int tokens_in_page = std::min<int>(
                static_cast<int>(pool.page_tokens()), sequence_length - processed);
            update_online(query, kvh, head_dim, scale, pool, pages[page_index],
                          0, tokens_in_page, destination, state);
            processed += tokens_in_page;
        }
        const float reciprocal = 1.0f / state.denominator;
        for (int d = 0; d < head_dim; ++d) destination[d] *= reciprocal;
    }
}

void cpu_gqa_decode_paged_parallel(
    const float* q, const CpuKvPagePool& pool,
    std::span<const CpuKvPageId> pages, float* output,
    int sequence_length, int q_heads, int kv_heads, int head_dim,
    CpuThreadPool& thread_pool, CpuPagedAttentionOptions options,
    CpuPagedAttentionStats* stats) {
    if (options.page_tile == 0) throw std::invalid_argument("CPU attention page tile must be positive");
    const size_t required_pages =
        (static_cast<size_t>(sequence_length) + pool.page_tokens() - 1) /
        pool.page_tokens();
    const size_t tiles = (required_pages + options.page_tile - 1) / options.page_tile;
    if (static_cast<size_t>(sequence_length) < options.parallel_threshold ||
        thread_pool.size() == 0 || tiles == 0) {
        cpu_gqa_decode_paged(q, pool, pages, output, sequence_length,
                             q_heads, kv_heads, head_dim);
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
                update_online(query, kvh, head_dim, scale, pool, pages[page_index],
                              0, tokens_in_page, accumulator, state);
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
        const float reciprocal = 1.0f / combined.denominator;
        for (int d = 0; d < head_dim; ++d) destination[d] *= reciprocal;
    }
    if (stats) *stats = {task_count, tiles, true};
}

} // namespace lfm
