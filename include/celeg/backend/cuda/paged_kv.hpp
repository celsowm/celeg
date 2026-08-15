#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy.hpp"
#include "celeg/runtime/cache/kv_page_allocator.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/runtime/cache/page_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace celeg {

class PhysicalPagedKvCache final : public IKvPageAllocator {
public:
    PhysicalPagedKvCache(size_t page_count,
                         int page_tokens,
                         int max_context,
                         KvCacheMode mode,
                         const ExecutionTopology& shape,
                         const CompiledModelProgram& program);
    ~PhysicalPagedKvCache() = default;

    PhysicalPagedKvCache(const PhysicalPagedKvCache&) = delete;
    PhysicalPagedKvCache& operator=(const PhysicalPagedKvCache&) = delete;

    std::optional<std::vector<uint32_t>> allocate_tokens(size_t token_count) override;
    bool retain(const std::vector<uint32_t>& pages) override;
    void release(const std::vector<uint32_t>& pages) override;
    uint32_t ref_count(uint32_t page) const;
    bool is_shared(uint32_t page) const { return ref_count(page) > 1; }

    std::optional<uint32_t> clone_page(uint32_t source);
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
    size_t page_vector_elements() const { return layout_.page_vector_elements(); }
    size_t page_scale_elements() const { return layout_.page_scale_elements(); }
    size_t layer_vector_offset(int attention_slot) const {
        return layout_.layer_vector_offset(attention_slot);
    }
    size_t layer_scale_offset(int attention_slot) const {
        return layout_.layer_scale_offset(attention_slot);
    }
    int layer_kv_width(int attention_slot) const {
        return layout_.layers.at(static_cast<size_t>(attention_slot)).kv_width;
    }
    int layer_kv_heads(int attention_slot) const {
        return layout_.layers.at(static_cast<size_t>(attention_slot)).kv_heads;
    }
    bool layer_is_latent(int attention_slot) const {
        return layout_.layers.at(static_cast<size_t>(attention_slot)).latent;
    }
    int layer_latent_rank(int attention_slot) const {
        return layout_.layers.at(static_cast<size_t>(attention_slot)).latent_rank;
    }
    int layer_rotary_width(int attention_slot) const {
        return layout_.layers.at(static_cast<size_t>(attention_slot)).rotary_width;
    }
    int attention_slot(int model_layer) const {
        if (model_layer < 0 ||
            model_layer >= static_cast<int>(attention_slot_for_layer_.size())) {
            return -1;
        }
        return attention_slot_for_layer_[static_cast<size_t>(model_layer)];
    }

private:
    int page_tokens_ = 0;
    int max_pages_per_request_ = 0;
    int attention_layer_count_ = 0;
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

}
