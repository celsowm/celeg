#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/runtime/cache/page_layout.hpp"
#include "celeg/runtime/checked_math.hpp"

#include <algorithm>
#include <stdexcept>

namespace celeg {

PhysicalPagedKvCache::PhysicalPagedKvCache(size_t page_count,
                                           int page_tokens,
                                           int max_context,
                                           KvCacheMode mode,
                                           const ExecutionTopology& shape,
                                           const CompiledModelProgram& program)
    : page_tokens_(page_tokens),
      mode_(mode),
      attention_layer_count_(shape.attention_layer_count),
      attention_slot_for_layer_(shape.attention_slot_for_layer),
      layout_(page_tokens, shape, program),
      allocator_(page_count, page_tokens) {
    if (max_context <= 0) throw std::invalid_argument("paged KV max_context must be positive");
    if (mode_ == KvCacheMode::Int8) {
        for (const auto& layer : layout_.layers) {
            if (layer.latent) {
                throw std::invalid_argument(
                    "CUDA INT8 paged state does not support latent attention");
            }
        }
    }
    max_pages_per_request_ =
        (max_context + page_tokens_ - 1) / page_tokens_;

    const size_t vectors = checked_multiply(
        page_count, layout_.page_vector_elements(),
        "paged KV vector allocation overflow");
    if (mode_ == KvCacheMode::Int8) {
        key_int8_.reset(vectors);
        value_int8_.reset(vectors);
        const size_t scales = checked_multiply(
            page_count, layout_.page_scale_elements(),
            "paged KV scale allocation overflow");
        key_scales_.reset(scales);
        value_scales_.reset(scales);
    } else {
        key_bf16_.reset(vectors);
        value_bf16_.reset(vectors);
    }
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
        const size_t source_vector = layout_.page_vector_offset(source);
        const size_t target_vector = layout_.page_vector_offset(target);
        // A page is laid out [layer][token][kv_width], so copying only a flat
        // prefix would skip later layers. Copy each layer's initialized token
        // region. The unused suffix is intentionally left untouched because
        // attention never reads beyond the request position and every future
        // slot is overwritten before it becomes visible.
        if (mode_ == KvCacheMode::Int8) {
            const size_t source_scale = layout_.page_scale_offset(source);
            const size_t target_scale = layout_.page_scale_offset(target);
            for (int layer = 0; layer < attention_layer_count_; ++layer) {
                const size_t layer_vector = layout_.layer_vector_offset(layer);
                const size_t count = layout_.layer_vector_count(layer, used_tokens);
                CELEG_CUDA(cudaMemcpy(key_int8_.data() + target_vector + layer_vector,
                                    key_int8_.data() + source_vector + layer_vector,
                                    count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(value_int8_.data() + target_vector + layer_vector,
                                    value_int8_.data() + source_vector + layer_vector,
                                    count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
                const size_t layer_scale = layout_.layer_scale_offset(layer);
                const size_t scale_count = layout_.layer_scale_count(layer, used_tokens);
                CELEG_CUDA(cudaMemcpy(key_scales_.data() + target_scale + layer_scale,
                                    key_scales_.data() + source_scale + layer_scale,
                                    scale_count * sizeof(float), cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(value_scales_.data() + target_scale + layer_scale,
                                    value_scales_.data() + source_scale + layer_scale,
                                    scale_count * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        } else {
            for (int layer = 0; layer < attention_layer_count_; ++layer) {
                const size_t layer_vector = layout_.layer_vector_offset(layer);
                const size_t count = layout_.layer_vector_count(layer, used_tokens);
                CELEG_CUDA(cudaMemcpy(key_bf16_.data() + target_vector + layer_vector,
                                    key_bf16_.data() + source_vector + layer_vector,
                                    count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
                CELEG_CUDA(cudaMemcpy(value_bf16_.data() + target_vector + layer_vector,
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

size_t PhysicalPagedKvCache::memory_bytes() const {
    return key_bf16_.bytes() + value_bf16_.bytes() +
           key_int8_.bytes() + value_int8_.bytes() +
           key_scales_.bytes() + value_scales_.bytes();
}

} // namespace celeg
