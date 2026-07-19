#pragma once

#include "lfm/cuda_utils.cuh"
#include "lfm/runtime_types.hpp"
#include "lfm/concurrent_policy.hpp"
#include "lfm/kv_page_allocator.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/page_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace lfm {

// Physical, reference-counted KV page arena shared by all requests in one
// ConcurrentEngine. A page contains the same token range for every attention
// layer of the variant, which makes a request page table valid across the
// entire model.
class PhysicalPagedKvCache final : public IKvPageAllocator {
public:
    PhysicalPagedKvCache(size_t page_count,
                         int page_tokens,
                         int max_context,
                         KvCacheMode mode,
                         const ModelShape& shape);
    ~PhysicalPagedKvCache() = default;

    PhysicalPagedKvCache(const PhysicalPagedKvCache&) = delete;
    PhysicalPagedKvCache& operator=(const PhysicalPagedKvCache&) = delete;

    std::optional<std::vector<uint32_t>> allocate_tokens(size_t token_count) override;
    bool retain(const std::vector<uint32_t>& pages) override;
    void release(const std::vector<uint32_t>& pages) override;
    uint32_t ref_count(uint32_t page) const;
    bool is_shared(uint32_t page) const { return ref_count(page) > 1; }

    // Allocates a new page and copies all K/V layers from source. Used by
    // prefix-cache copy-on-write when the final cached page is partial.
    std::optional<uint32_t> clone_page(uint32_t source);
    // Copy only the initialized token prefix from a partial page. The unused
    // suffix is zeroed, reducing COW traffic for short shared prefixes.
    std::optional<uint32_t> clone_page_prefix(uint32_t source, int used_tokens) override;

    size_t total_pages() const override { return allocator_.total_pages(); }
    size_t free_pages() const { return allocator_.free_pages(); }
    size_t used_pages() const override { return allocator_.used_pages(); }
    int page_tokens() const override { return page_tokens_; }
    int max_pages_per_request() const { return max_pages_per_request_; }
    KvCacheMode mode() const { return mode_; }
    size_t memory_bytes() const;
    size_t page_bytes() const override {
        return total_pages() == 0 ? 0 : memory_bytes() / total_pages();
    }

    __nv_bfloat16* key_bf16() { return key_bf16_.data(); }
    __nv_bfloat16* value_bf16() { return value_bf16_.data(); }
    int8_t* key_int8() { return key_int8_.data(); }
    int8_t* value_int8() { return value_int8_.data(); }
    float* key_scales() { return key_scales_.data(); }
    float* value_scales() { return value_scales_.data(); }

    const __nv_bfloat16* key_bf16() const { return key_bf16_.data(); }
    const __nv_bfloat16* value_bf16() const { return value_bf16_.data(); }
    const int8_t* key_int8() const { return key_int8_.data(); }
    const int8_t* value_int8() const { return value_int8_.data(); }
    const float* key_scales() const { return key_scales_.data(); }
    const float* value_scales() const { return value_scales_.data(); }

    int attention_layers() const { return attention_layer_count_; }
    int kv_width() const { return kv_width_; }
    int kv_heads() const { return kv_heads_; }
    int head_dim() const { return head_dim_; }
    // Returns the attention slot for a given model layer index, or -1 if the
    // layer is a convolution layer.
    int attention_slot(int model_layer) const {
        if (model_layer < 0 ||
            model_layer >= static_cast<int>(attention_slot_for_layer_.size())) {
            return -1;
        }
        return attention_slot_for_layer_[static_cast<size_t>(model_layer)];
    }

private:
    size_t page_vector_elements() const;
    size_t page_scale_elements() const;

    int page_tokens_ = 0;
    int max_pages_per_request_ = 0;
    int attention_layer_count_ = 0;
    int kv_width_ = 0;
    int kv_heads_ = 0;
    int head_dim_ = 0;
    KvCacheMode mode_ = KvCacheMode::Bf16;
    std::vector<int> attention_slot_for_layer_;
    PageLayout layout_;
    PagedBlockPool allocator_;

    DeviceBuffer<__nv_bfloat16> key_bf16_;
    DeviceBuffer<__nv_bfloat16> value_bf16_;
    DeviceBuffer<int8_t> key_int8_;
    DeviceBuffer<int8_t> value_int8_;
    DeviceBuffer<float> key_scales_;
    DeviceBuffer<float> value_scales_;
};

} // namespace lfm
