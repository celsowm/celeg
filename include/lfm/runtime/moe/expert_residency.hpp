#pragma once

#include "lfm/backend/cuda/utils.cuh"
#include "lfm/runtime/moe/offload.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfm {

// ---------------------------------------------------------------------------
// Tier 2 host expert store.
//
// Owns host-resident BF16 storage for offloaded MoE experts and exposes
// device-visible pointers (via CUDA UVA) so the MoE FFN kernel can read a cold
// expert directly over PCIe without copying it to the GPU first.
//
// Two backends mirror the plan:
//   * ExpertHostMode::Mapped     -> cudaHostRegister a caller-owned range
//                                   (e.g. an mmap'd safetensors region) and
//                                   obtain a device pointer for it.
//   * ExpertHostMode::PinnedCopy -> cudaHostAlloc a pinned arena and copy the
//                                   expert bytes into it.
// Staged mode reuses PinnedCopy semantics for the host tier.
// ---------------------------------------------------------------------------
class HostExpertStore {
public:
    HostExpertStore() = default;
    ~HostExpertStore();

    HostExpertStore(const HostExpertStore&) = delete;
    HostExpertStore& operator=(const HostExpertStore&) = delete;
    HostExpertStore(HostExpertStore&&) noexcept;
    HostExpertStore& operator=(HostExpertStore&&) noexcept;

    // Registers a caller-owned host range and returns a device-visible pointer.
    // The caller must keep `host_ptr` valid for the lifetime of this store.
    // Used for ExpertHostMode::Mapped against mmap'd safetensors regions.
    const void* register_mapped(const void* host_ptr, std::size_t bytes);

    // Allocates a host buffer owned by this store, pins+registers it (mapped),
    // and returns a device-visible pointer. Used for ExpertHostMode::Mapped when
    // the source is not already a stable mmap'd region (e.g. a packed per-layer
    // arena assembled from separate safetensors tensors). The buffer is freed
    // with the store.
    const void* alloc_mapped(std::size_t bytes);

    // Copies `bytes` from `src` into a freshly pinned, mapped allocation and
    // returns a device-visible pointer to the copy. Used for PinnedCopy/Staged.
    const void* store_pinned_copy(const void* src, std::size_t bytes);

    std::size_t registered_bytes() const { return registered_bytes_; }
    std::size_t pinned_bytes() const { return pinned_bytes_; }

private:
    struct Registration {
        void* host_ptr = nullptr;   // page-aligned base actually registered
        bool owned = false;         // true when allocated by the store
        bool mapped = false;        // registered via cudaHostRegister (not cudaHostAlloc)
    };
    std::vector<Registration> registrations_;
    std::size_t registered_bytes_ = 0;
    std::size_t pinned_bytes_ = 0;
    void release();
};

// ---------------------------------------------------------------------------
// Tier 1 per-layer GPU expert cache + device pointer tables.
//
// Manages residency for one MoE layer. `capacity` GPU cache slots hold hot
// experts; every other expert is served from its host-mapped device pointer.
// Two device pointer arrays (gate_up_ptrs / down_ptrs, [num_experts]) are what
// the MoE FFN kernel dereferences: each entry points either at a GPU cache
// slot or at host-mapped memory.
// ---------------------------------------------------------------------------
class ExpertLayerCache {
public:
    // Byte size of one expert's gate_up and down tensors (BF16).
    ExpertLayerCache(int num_experts, int capacity,
                     std::size_t gate_up_bytes, std::size_t down_bytes);

    ExpertLayerCache(const ExpertLayerCache&) = delete;
    ExpertLayerCache& operator=(const ExpertLayerCache&) = delete;

    // Sets the host-mapped device pointers for every expert. These are the
    // fallback (slow-path) locations used when an expert is not cached.
    // `gate_up_host_dev` / `down_host_dev` are [num_experts] device-visible
    // pointers (from HostExpertStore). Must be called before use.
    void set_host_sources(const std::vector<const __nv_bfloat16*>& gate_up_host_dev,
                          const std::vector<const __nv_bfloat16*>& down_host_dev);

    // Promotes `expert` into cache `slot`, copying its weights from the host
    // source into the slot on `stream` and updating the device pointer table
    // once the copy is ordered. Evicts whatever occupied the slot. `score` is
    // the routing likelihood this promotion is based on (used to pick future
    // victims); the default marks the expert as low-priority so it is evicted
    // before any score-ranked expert.
    void promote(int expert, int slot, cudaStream_t stream,
                 float score = -1.0e30f);

    void promote(int expert, int slot, const __nv_bfloat16* gate_up_src,
                 const __nv_bfloat16* down_src, cudaStream_t stream,
                 float score = -1.0e30f);

    // Ensures `expert` is GPU-resident: if already cached, just refreshes its
    // LRU recency; otherwise promotes it into the lowest-score slot. `score` is
    // the routing likelihood of this expert (the router's per-expert score for
    // the current token); on-demand promotions (a token actually routed here)
    // pass the real score so the resident set converges to the highest-likelihood
    // experts. Returns true when a host->device promotion was performed.
    bool ensure_resident(int expert, cudaStream_t stream, float score = -1.0e30f);

    bool ensure_resident(int expert, const __nv_bfloat16* gate_up_src,
                         const __nv_bfloat16* down_src, cudaStream_t stream,
                         float score = -1.0e30f);

    // Speculatively promotes up to `n` of the most-recently-used *not-yet-
    // resident* experts (ranked by last access tick) so they are GPU-resident
    // before the next decode step. Runs on `stream` (the offload transfer
    // stream) and overlaps the current token's FFN compute. Returns the number
    // of experts actually promoted.
    int prefetch(int n, cudaStream_t stream);

    // Router-guided speculative prefetch: `ranked` is a caller-supplied list of
    // experts in descending order of predicted routing likelihood, `scores` the
    // matching per-expert routing scores. Promotes at most `n` candidates that
    // are not already GPU-resident AND whose likelihood exceeds the lowest-score
    // resident (so speculative promotions never displace a more-likely expert).
    // Returns the number promoted. This targets the experts most likely to be
    // selected next, which the pure recency-based `prefetch` cannot see.
    int prefetch_list(const std::vector<int>& ranked,
                      const std::vector<float>& scores, int n,
                      cudaStream_t stream);

    // Marks `expert` as recently used (updates LRU recency) without copying.
    // No-op if the expert is host-resident. The score overload also refreshes
    // the expert's routing-likelihood priority so a hit keeps it resident.
    void touch(int expert);
    void touch(int expert, float score);

    // Residency hit-rate instrumentation (decode path). `record_hit` /
    // `record_miss` are called by the model when it routes to a resident /
    // host-only expert; `hit_rate` returns hits/(hits+misses) in [0,1].
    void record_hit() { ++hits_; }
    void record_miss() { ++misses_; }
    double hit_rate() const {
        const uint64_t total = hits_ + misses_;
        return total == 0 ? 0.0 : static_cast<double>(hits_) / total;
    }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }

    // Marks a cache slot empty and repoints the expert it held back at its
    // host source. No-op if the slot is empty.
    void evict(int slot, cudaStream_t stream);

    // Convenience: fill the cache with experts [0, capacity) (or a supplied
    // list) starting from a cold state. Blocks are ordered on `stream`.
    void seed(const std::vector<int>& experts, cudaStream_t stream);

    // Device pointer tables for the MoE FFN kernel.
    const __nv_bfloat16* const* gate_up_ptrs() const { return gate_up_ptrs_dev_.data(); }
    const __nv_bfloat16* const* down_ptrs() const { return down_ptrs_dev_.data(); }

    const __nv_bfloat16* expert_gate_up_dev(int expert) const;
    const __nv_bfloat16* expert_down_dev(int expert) const;

    // Device-side residency check: reads sel_dev and expert_slot_dev_ on the
    // GPU, outputs a compact list of cold experts that need promotion.
    // Returns the number of cold experts. cold_host receives the cold expert
    // indices; cold_scores_host receives per-expert scores (E floats) when
    // route_scores_dev is non-null.
    int resolve_on_device(const int* sel_dev, const float* route_scores_dev,
                          int rows, int K, cudaStream_t stream,
                          std::vector<int>& cold_host,
                          std::vector<float>& cold_scores_host);

    void sync_residency_tables(cudaStream_t stream = nullptr);

    int num_experts() const { return num_experts_; }
    int capacity() const { return capacity_; }
    // Expert currently occupying `slot`, or -1 if empty.
    int slot_expert(int slot) const { return slots_.at(static_cast<size_t>(slot)).expert; }
    // Cache slot holding `expert`, or -1 if it is host-resident.
    int expert_slot(int expert) const { return expert_slot_.at(static_cast<size_t>(expert)); }
    bool resident(int expert) const { return expert_slot(expert) >= 0; }

    // Selects the eviction order for occupied slots. `Lru` (default) evicts the
    // least-recently-used; `Score` evicts the lowest routing-likelihood (uses
    // last_score_, populated from on-demand hits/promotions). The policy is
    // per-cache; experiments show LRU dominates Score on LFM2-8B-A1B.
    void set_policy(ExpertCachePolicy p) { policy_ = p; }
    ExpertCachePolicy policy() const { return policy_; }

    // Sentry routing-likelihood score meaning "no signal yet" (seeded experts or
    // paths without router scores). Lowest possible priority, so any
    // router-ranked promotion displaces an unseen resident, and the resident set
    // converges to the highest-likelihood experts as soon as the first token
    // routes through the layer.
    static constexpr float kUnseen = -1.0e30f;

private:
    struct Slot {
        DeviceBuffer<__nv_bfloat16> gate_up;
        DeviceBuffer<__nv_bfloat16> down;
        int expert = -1;
        uint64_t last_used = 0;   // LRU tick of last access
    };

    // Picks the victim slot for a new promotion: the least-recently-used
    // occupied slot, or the first free slot if any. Caller must hold capacity>0.
    int choose_victim() const;

    void publish_pointer(int expert, const __nv_bfloat16* gate_up,
                         const __nv_bfloat16* down, cudaStream_t stream);
    void publish_pointer_immediate(int expert,
                                   const __nv_bfloat16* gate_up,
                                   const __nv_bfloat16* down);

    int num_experts_ = 0;
    int capacity_ = 0;
    std::size_t gate_up_bytes_ = 0;
    std::size_t down_bytes_ = 0;
    ExpertCachePolicy policy_ = ExpertCachePolicy::Lru;
    uint64_t lru_tick_ = 0;       // monotonically increasing access counter

    std::vector<Slot> slots_;
    std::vector<int> expert_slot_;              // [E] -> slot or -1
    std::vector<uint64_t> last_used_;           // [E] per-expert recency tick
    // [E] per-expert routing-likelihood priority. Lower score = evict first.
    // Seeded/unseen experts default to kUnseen (-1e30) so any router-ranked
    // promotion displaces them; on-demand hits/promotions refresh it with the
    // current token's router score, so the resident set converges to the
    // highest-likelihood experts.
    std::vector<float> last_score_;

    uint64_t hits_ = 0;
    uint64_t misses_ = 0;

    std::vector<const __nv_bfloat16*> gate_up_host_dev_;  // [E] host UVA ptrs
    std::vector<const __nv_bfloat16*> down_host_dev_;      // [E]
    std::vector<const __nv_bfloat16*> gate_up_ptrs_host_;  // [E] current table
    std::vector<const __nv_bfloat16*> down_ptrs_host_;     // [E] current table

    // Device-resident pointer tables consumed by the kernel.
    DeviceBuffer<const __nv_bfloat16*> gate_up_ptrs_dev_;  // [E]
    DeviceBuffer<const __nv_bfloat16*> down_ptrs_dev_;      // [E]

    // Device-side residency check state.
    DeviceBuffer<int> expert_slot_dev_;   // [E] mirror of expert_slot_
    DeviceBuffer<int> cold_flags_dev_;    // [E] temporary flags for deduplication
    DeviceBuffer<int> cold_list_dev_;     // [E] output: cold expert indices
    DeviceBuffer<int> cold_count_dev_;    // [1] output: number of cold experts

    void sync_expert_slot_to_device(cudaStream_t stream = nullptr);
};

} // namespace lfm
