#pragma once

#include "celeg/backend/cuda/utils.cuh"
#include "celeg/backend/cuda/moe/offload.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

class HostExpertStore {
public:
    HostExpertStore() = default;
    ~HostExpertStore();

    HostExpertStore(const HostExpertStore&) = delete;
    HostExpertStore& operator=(const HostExpertStore&) = delete;
    HostExpertStore(HostExpertStore&&) noexcept;
    HostExpertStore& operator=(HostExpertStore&&) noexcept;

    const void* register_mapped(const void* host_ptr, std::size_t bytes);
    const void* alloc_mapped(std::size_t bytes);
    const void* store_pinned_copy(const void* src, std::size_t bytes);

    std::size_t registered_bytes() const { return registered_bytes_; }
    std::size_t pinned_bytes() const { return pinned_bytes_; }

private:
    struct Registration {
        void* host_ptr = nullptr;
        bool owned = false;
        bool mapped = false;
    };

    std::vector<Registration> registrations_;
    std::size_t registered_bytes_ = 0;
    std::size_t pinned_bytes_ = 0;
    void release();
};

class ExpertLayerCache {
public:
    ExpertLayerCache(int num_experts, int capacity,
                     std::size_t gate_up_bytes, std::size_t down_bytes);

    ExpertLayerCache(const ExpertLayerCache&) = delete;
    ExpertLayerCache& operator=(const ExpertLayerCache&) = delete;

    void set_host_sources(const std::vector<const __nv_bfloat16*>& gate_up_host_dev,
                          const std::vector<const __nv_bfloat16*>& down_host_dev);

    void promote(int expert, int slot, cudaStream_t stream,
                 float score = -1.0e30f);
    void promote(int expert, int slot, const __nv_bfloat16* gate_up_src,
                 const __nv_bfloat16* down_src, cudaStream_t stream,
                 float score = -1.0e30f);

    bool ensure_resident(int expert, cudaStream_t stream,
                         float score = -1.0e30f);
    bool ensure_resident(int expert, const __nv_bfloat16* gate_up_src,
                         const __nv_bfloat16* down_src, cudaStream_t stream,
                         float score = -1.0e30f);

    int prefetch(int n, cudaStream_t stream);
    int prefetch_list(const std::vector<int>& ranked,
                      const std::vector<float>& scores, int n,
                      cudaStream_t stream);

    void touch(int expert);
    void touch(int expert, float score);

    void record_hit() { ++hits_; }
    void record_miss() { ++misses_; }
    double hit_rate() const {
        const uint64_t total = hits_ + misses_;
        return total == 0 ? 0.0 : static_cast<double>(hits_) / total;
    }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }

    void evict(int slot, cudaStream_t stream);

    // Seeded experts remain probationary by default. A seed derived from a
    // persisted usage profile may be protected immediately.
    void seed(const std::vector<int>& experts, cudaStream_t stream,
              bool protect = false);

    const __nv_bfloat16* const* gate_up_ptrs() const {
        return gate_up_ptrs_dev_.data();
    }
    const __nv_bfloat16* const* down_ptrs() const {
        return down_ptrs_dev_.data();
    }

    const __nv_bfloat16* expert_gate_up_dev(int expert) const;
    const __nv_bfloat16* expert_down_dev(int expert) const;

    int resolve_on_device(const int* sel_dev, const float* route_scores_dev,
                          int rows, int K, cudaStream_t stream,
                          std::vector<int>& cold_host,
                          std::vector<float>& cold_scores_host);

    void sync_residency_tables(cudaStream_t stream = nullptr);

    int num_experts() const { return num_experts_; }
    int capacity() const { return capacity_; }
    int protected_capacity() const { return protected_capacity_; }
    int probation_capacity() const { return capacity_ - protected_capacity_; }
    int protected_entries() const { return protected_count_; }

    int slot_expert(int slot) const {
        return slots_.at(static_cast<size_t>(slot)).expert;
    }
    int expert_slot(int expert) const {
        return expert_slot_.at(static_cast<size_t>(expert));
    }
    bool resident(int expert) const { return expert_slot(expert) >= 0; }
    bool protected_resident(int expert) const {
        const int slot = expert_slot(expert);
        return slot >= 0 && slots_.at(static_cast<size_t>(slot)).protected_entry;
    }

    void set_policy(ExpertCachePolicy policy);
    ExpertCachePolicy policy() const { return policy_; }

    static constexpr float kUnseen = -1.0e30f;

private:
    struct Slot {
        DeviceBuffer<__nv_bfloat16> gate_up;
        DeviceBuffer<__nv_bfloat16> down;
        int expert = -1;
        uint64_t last_used = 0;
        uint32_t observed_accesses = 0;
        bool protected_entry = false;
    };

    int choose_victim(bool speculative = false) const;
    int choose_coldest_protected(int except_slot = -1) const;
    double priority(int expert) const;
    void mark_probation(int slot);
    void promote_to_protected(int slot);

    void publish_pointer(int expert, const __nv_bfloat16* gate_up,
                         const __nv_bfloat16* down, cudaStream_t stream);
    void publish_pointer_immediate(int expert,
                                   const __nv_bfloat16* gate_up,
                                   const __nv_bfloat16* down);

    int num_experts_ = 0;
    int capacity_ = 0;
    int protected_capacity_ = 0;
    int protected_count_ = 0;
    std::size_t gate_up_bytes_ = 0;
    std::size_t down_bytes_ = 0;
    ExpertCachePolicy policy_ = ExpertCachePolicy::Lru;
    uint64_t lru_tick_ = 0;

    std::vector<Slot> slots_;
    std::vector<int> expert_slot_;
    std::vector<uint64_t> last_used_;
    std::vector<float> last_score_;

    uint64_t hits_ = 0;
    uint64_t misses_ = 0;

    std::vector<const __nv_bfloat16*> gate_up_host_dev_;
    std::vector<const __nv_bfloat16*> down_host_dev_;
    std::vector<const __nv_bfloat16*> gate_up_ptrs_host_;
    std::vector<const __nv_bfloat16*> down_ptrs_host_;

    DeviceBuffer<const __nv_bfloat16*> gate_up_ptrs_dev_;
    DeviceBuffer<const __nv_bfloat16*> down_ptrs_dev_;

    DeviceBuffer<int> expert_slot_dev_;
    DeviceBuffer<int> cold_flags_dev_;
    DeviceBuffer<int> cold_list_dev_;
    DeviceBuffer<int> cold_count_dev_;

    void sync_expert_slot_to_device(cudaStream_t stream = nullptr);
};

} // namespace celeg
