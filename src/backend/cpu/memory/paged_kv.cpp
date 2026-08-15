#include "celeg/backend/cpu/paged_kv.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/backend/cpu/isa.hpp"
#include "celeg/runtime/checked_math.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {

namespace {

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
    const size_t allocation_bytes = checked_round_up(
        page_bytes_, alignment, "CPU KV aligned allocation overflow");
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

}
