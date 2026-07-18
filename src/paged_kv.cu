#include "lfm/paged_kv.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lfm {

namespace {

size_t checked_mul(size_t a, size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(what);
    }
    return a * b;
}

} // namespace

PhysicalPagedKvCache::PhysicalPagedKvCache(size_t page_count,
                                           int page_tokens,
                                           int max_context,
                                           KvCacheMode mode)
    : page_tokens_(page_tokens), mode_(mode), allocator_(page_count, page_tokens) {
    if (max_context <= 0) throw std::invalid_argument("paged KV max_context must be positive");
    max_pages_per_request_ =
        (max_context + page_tokens_ - 1) / page_tokens_;

    const size_t vectors = checked_mul(page_count, page_vector_elements(),
                                       "paged KV vector allocation overflow");
    if (mode_ == KvCacheMode::Int8) {
        key_int8_.reset(vectors);
        value_int8_.reset(vectors);
        const size_t scales = checked_mul(page_count, page_scale_elements(),
                                          "paged KV scale allocation overflow");
        key_scales_.reset(scales);
        value_scales_.reset(scales);
    } else {
        key_bf16_.reset(vectors);
        value_bf16_.reset(vectors);
    }
}

size_t PhysicalPagedKvCache::page_vector_elements() const {
    return checked_mul(
        checked_mul(static_cast<size_t>(attention_layers),
                    static_cast<size_t>(page_tokens_),
                    "paged KV page layer/token overflow"),
        static_cast<size_t>(LfmConfig::kv_width),
        "paged KV page vector overflow");
}

size_t PhysicalPagedKvCache::page_scale_elements() const {
    return checked_mul(
        checked_mul(static_cast<size_t>(attention_layers),
                    static_cast<size_t>(page_tokens_),
                    "paged KV scale layer/token overflow"),
        static_cast<size_t>(LfmConfig::kv_heads),
        "paged KV page scale overflow");
}

std::optional<std::vector<uint32_t>>
PhysicalPagedKvCache::allocate_tokens(size_t token_count) {
    return allocator_.allocate_tokens(token_count);
}

bool PhysicalPagedKvCache::retain(const std::vector<uint32_t>& pages) {
    return allocator_.retain(pages);
}

void PhysicalPagedKvCache::release(const std::vector<uint32_t>& pages) {
    allocator_.release(pages);
}

uint32_t PhysicalPagedKvCache::ref_count(uint32_t page) const {
    return allocator_.ref_count(page);
}

std::optional<uint32_t> PhysicalPagedKvCache::clone_page(uint32_t source) {
    return clone_page_prefix(source, page_tokens_);
}

std::optional<uint32_t> PhysicalPagedKvCache::clone_page_prefix(
    uint32_t source, int used_tokens) {
    if (source >= total_pages() || ref_count(source) == 0) {
        throw std::invalid_argument("cannot clone an invalid or free KV page");
    }
    if (used_tokens <= 0 || used_tokens > page_tokens_) {
        throw std::invalid_argument("partial KV clone used_tokens is out of range");
    }
    auto allocated = allocator_.allocate_tokens(static_cast<size_t>(page_tokens_));
    if (!allocated || allocated->size() != 1) return std::nullopt;
    const uint32_t target = allocated->front();
    try {
        const size_t full_vector_elements = page_vector_elements();
        const size_t source_vector = static_cast<size_t>(source) * full_vector_elements;
        const size_t target_vector = static_cast<size_t>(target) * full_vector_elements;
        // A page is laid out [layer][token][kv_width], so copying only a flat
        // prefix would skip later layers. Copy each layer's initialized token
        // region. The unused suffix is intentionally left untouched because
        // attention never reads beyond the request position and every future
        // slot is overwritten before it becomes visible.
        if (mode_ == KvCacheMode::Int8) {
            const size_t full_scale_elements = page_scale_elements();
            const size_t source_scale = static_cast<size_t>(source) * full_scale_elements;
            const size_t target_scale = static_cast<size_t>(target) * full_scale_elements;
            for (int layer = 0; layer < attention_layers; ++layer) {
                const size_t layer_vector = static_cast<size_t>(layer) *
                    page_tokens_ * LfmConfig::kv_width;
                const size_t count = static_cast<size_t>(used_tokens) *
                    LfmConfig::kv_width;
                LFM_CUDA(cudaMemcpy(key_int8_.data() + target_vector + layer_vector,
                                    key_int8_.data() + source_vector + layer_vector,
                                    count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
                LFM_CUDA(cudaMemcpy(value_int8_.data() + target_vector + layer_vector,
                                    value_int8_.data() + source_vector + layer_vector,
                                    count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
                const size_t layer_scale = static_cast<size_t>(layer) *
                    page_tokens_ * LfmConfig::kv_heads;
                const size_t scale_count = static_cast<size_t>(used_tokens) *
                    LfmConfig::kv_heads;
                LFM_CUDA(cudaMemcpy(key_scales_.data() + target_scale + layer_scale,
                                    key_scales_.data() + source_scale + layer_scale,
                                    scale_count * sizeof(float), cudaMemcpyDeviceToDevice));
                LFM_CUDA(cudaMemcpy(value_scales_.data() + target_scale + layer_scale,
                                    value_scales_.data() + source_scale + layer_scale,
                                    scale_count * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        } else {
            for (int layer = 0; layer < attention_layers; ++layer) {
                const size_t layer_vector = static_cast<size_t>(layer) *
                    page_tokens_ * LfmConfig::kv_width;
                const size_t count = static_cast<size_t>(used_tokens) *
                    LfmConfig::kv_width;
                LFM_CUDA(cudaMemcpy(key_bf16_.data() + target_vector + layer_vector,
                                    key_bf16_.data() + source_vector + layer_vector,
                                    count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
                LFM_CUDA(cudaMemcpy(value_bf16_.data() + target_vector + layer_vector,
                                    value_bf16_.data() + source_vector + layer_vector,
                                    count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            }
        }
    } catch (...) {
        allocator_.release(std::vector<uint32_t>{target});
        throw;
    }
    return target;
}

int PhysicalPagedKvCache::attention_slot(int model_layer) {
    switch (model_layer) {
        case 2: return 0;
        case 4: return 1;
        case 6: return 2;
        case 8: return 3;
        case 10: return 4;
        case 12: return 5;
        default: return -1;
    }
}

size_t PhysicalPagedKvCache::memory_bytes() const {
    return key_bf16_.bytes() + value_bf16_.bytes() +
           key_int8_.bytes() + value_int8_.bytes() +
           key_scales_.bytes() + value_scales_.bytes();
}

} // namespace lfm
