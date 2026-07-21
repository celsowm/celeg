#include "lfm/expert_residency.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfm {

namespace {

// Rounds `value` down / up to a page-aligned boundary. cudaHostRegister
// requires page-aligned ranges; we register the enclosing aligned range and
// offset the returned device pointer back to the requested address.
constexpr std::size_t kPageSize = 4096;

std::uintptr_t align_down(std::uintptr_t v) { return v & ~(kPageSize - 1); }
std::size_t align_up(std::size_t v) {
    return (v + kPageSize - 1) & ~(kPageSize - 1);
}

// Device kernel: for each selected expert, check if it is GPU-resident.
// Outputs a compact list of cold experts that need promotion.
//   sel_dev[i]          = selected expert index (rows * K total)
//   expert_slot_dev[e]  = cache slot for expert e, or -1 if host-resident
//   cold_list_dev[out]  = compact list of cold expert indices
//   cold_count_dev[out] = number of cold experts (atomic, initialized to 0)
__global__ void resolve_residency_kernel(const int* sel_dev,
                                         const int* expert_slot_dev,
                                         int* cold_list_dev,
                                         int* cold_count_dev,
                                         int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int e = sel_dev[idx];
    if (expert_slot_dev[e] < 0) {
        const int pos = atomicAdd(cold_count_dev, 1);
        cold_list_dev[pos] = e;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// HostExpertStore
// ---------------------------------------------------------------------------

HostExpertStore::HostExpertStore(HostExpertStore&& other) noexcept {
    *this = std::move(other);
}

HostExpertStore& HostExpertStore::operator=(HostExpertStore&& other) noexcept {
    if (this != &other) {
        release();
        registrations_ = std::move(other.registrations_);
        registered_bytes_ = std::exchange(other.registered_bytes_, 0);
        pinned_bytes_ = std::exchange(other.pinned_bytes_, 0);
        other.registrations_.clear();
    }
    return *this;
}

HostExpertStore::~HostExpertStore() { release(); }

void HostExpertStore::release() {
    for (Registration& reg : registrations_) {
        if (reg.host_ptr == nullptr) continue;
        if (reg.mapped) {
            cudaHostUnregister(reg.host_ptr);
        } else {
            cudaFreeHost(reg.host_ptr);
        }
        reg.host_ptr = nullptr;
    }
    registrations_.clear();
    registered_bytes_ = 0;
    pinned_bytes_ = 0;
}

const void* HostExpertStore::register_mapped(const void* host_ptr,
                                             std::size_t bytes) {
    if (host_ptr == nullptr || bytes == 0) {
        throw std::invalid_argument("register_mapped: empty range");
    }
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(host_ptr);
    const std::uintptr_t base = align_down(addr);
    const std::size_t offset = static_cast<std::size_t>(addr - base);
    const std::size_t aligned_bytes = align_up(offset + bytes);

    void* base_ptr = reinterpret_cast<void*>(base);
    LFM_CUDA(cudaHostRegister(base_ptr, aligned_bytes,
                              cudaHostRegisterMapped | cudaHostRegisterPortable));
    void* dev_base = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev_base, base_ptr, 0));

    registrations_.push_back(Registration{base_ptr, /*owned=*/false});
    registered_bytes_ += aligned_bytes;
    return reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(dev_base) + offset);
}

const void* HostExpertStore::store_pinned_copy(const void* src,
                                                std::size_t bytes) {
    if (src == nullptr || bytes == 0) {
        throw std::invalid_argument("store_pinned_copy: empty range");
    }
    void* host = nullptr;
    LFM_CUDA(cudaHostAlloc(&host, bytes,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(host, src, bytes);
    void* dev = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev, host, 0));

    registrations_.push_back(Registration{host, /*owned=*/true, /*mapped=*/false});
    pinned_bytes_ += bytes;
    return dev;
}

const void* HostExpertStore::alloc_mapped(std::size_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument("alloc_mapped: empty range");
    }
    // A single pinned, mapped host allocation serving as the whole-layer arena.
    // cudaHostAlloc with Mapped|Portable yields a device-visible pointer, so no
    // separate cudaHostRegister call is needed (registering CUDA-allocated
    // memory is invalid).
    void* host = nullptr;
    LFM_CUDA(cudaHostAlloc(&host, bytes,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    void* dev = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev, host, 0));
    registrations_.push_back(Registration{host, /*owned=*/true, /*mapped=*/false});
    pinned_bytes_ += bytes;
    return dev;
}

// ---------------------------------------------------------------------------
// ExpertLayerCache
// ---------------------------------------------------------------------------

ExpertLayerCache::ExpertLayerCache(int num_experts, int capacity,
                                   std::size_t gate_up_bytes,
                                   std::size_t down_bytes)
    : num_experts_(num_experts),
      capacity_(capacity),
      gate_up_bytes_(gate_up_bytes),
      down_bytes_(down_bytes) {
    if (num_experts <= 0 || capacity < 0 || capacity > num_experts) {
        throw std::invalid_argument("invalid ExpertLayerCache dimensions");
    }
    if (gate_up_bytes == 0 || down_bytes == 0 ||
        gate_up_bytes % sizeof(__nv_bfloat16) != 0 ||
        down_bytes % sizeof(__nv_bfloat16) != 0) {
        throw std::invalid_argument("invalid ExpertLayerCache byte sizes");
    }
    slots_.resize(static_cast<size_t>(capacity));
    for (Slot& slot : slots_) {
        slot.gate_up.reset(gate_up_bytes / sizeof(__nv_bfloat16));
        slot.down.reset(down_bytes / sizeof(__nv_bfloat16));
        slot.expert = -1;
    }
    expert_slot_.assign(static_cast<size_t>(num_experts), -1);
    last_used_.assign(static_cast<size_t>(num_experts), 0);
    last_score_.assign(static_cast<size_t>(num_experts), kUnseen);
    gate_up_host_dev_.assign(static_cast<size_t>(num_experts), nullptr);
    down_host_dev_.assign(static_cast<size_t>(num_experts), nullptr);
    gate_up_ptrs_dev_.reset(static_cast<size_t>(num_experts));
    down_ptrs_dev_.reset(static_cast<size_t>(num_experts));

    // Device-side residency check buffers.
    expert_slot_dev_.reset(static_cast<size_t>(num_experts));
    LFM_CUDA(cudaMemcpy(expert_slot_dev_.data(), expert_slot_.data(),
                        static_cast<size_t>(num_experts) * sizeof(int),
                        cudaMemcpyHostToDevice));
    cold_list_dev_.reset(static_cast<size_t>(num_experts));
    cold_count_dev_.reset(1);
}

void ExpertLayerCache::set_host_sources(
    const std::vector<const __nv_bfloat16*>& gate_up_host_dev,
    const std::vector<const __nv_bfloat16*>& down_host_dev) {
    if (static_cast<int>(gate_up_host_dev.size()) != num_experts_ ||
        static_cast<int>(down_host_dev.size()) != num_experts_) {
        throw std::invalid_argument("set_host_sources: size mismatch");
    }
    gate_up_host_dev_ = gate_up_host_dev;
    down_host_dev_ = down_host_dev;
    // Every expert starts host-resident: publish host pointers into the table.
    LFM_CUDA(cudaMemcpy(gate_up_ptrs_dev_.data(), gate_up_host_dev_.data(),
                        static_cast<size_t>(num_experts_) * sizeof(const __nv_bfloat16*),
                        cudaMemcpyHostToDevice));
    LFM_CUDA(cudaMemcpy(down_ptrs_dev_.data(), down_host_dev_.data(),
                        static_cast<size_t>(num_experts_) * sizeof(const __nv_bfloat16*),
                        cudaMemcpyHostToDevice));
    sync_expert_slot_to_device();
}

void ExpertLayerCache::sync_expert_slot_to_device() {
    LFM_CUDA(cudaMemcpy(expert_slot_dev_.data(), expert_slot_.data(),
                        static_cast<size_t>(num_experts_) * sizeof(int),
                        cudaMemcpyHostToDevice));
}

int ExpertLayerCache::resolve_on_device(
    const int* sel_dev, const float* route_scores_dev,
    int rows, int K, cudaStream_t stream,
    std::vector<int>& cold_host,
    std::vector<float>& cold_scores_host) {
    const int total = rows * K;
    if (total == 0) return 0;

    // Reset the cold count on device.
    const int zero = 0;
    LFM_CUDA(cudaMemcpyAsync(cold_count_dev_.data(), &zero, sizeof(int),
                             cudaMemcpyHostToDevice, stream));

    // Launch the residency check kernel.
    const int block = 256;
    const int grid = (total + block - 1) / block;
    resolve_residency_kernel<<<grid, block, 0, stream>>>(
        sel_dev, expert_slot_dev_.data(),
        cold_list_dev_.data(), cold_count_dev_.data(), total);

    // Read back the cold count.
    int cold_count = 0;
    LFM_CUDA(cudaMemcpyAsync(&cold_count, cold_count_dev_.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
    LFM_CUDA(cudaStreamSynchronize(stream));

    if (cold_count == 0) {
        // All selected experts are GPU-resident. Touch them on host.
        // We still need the selection to update LRU/score, but only for
        // resident experts. Read the selection back for the small K*rows set.
        cold_host.resize(static_cast<size_t>(total));
        LFM_CUDA(cudaMemcpyAsync(cold_host.data(), sel_dev,
                                 static_cast<size_t>(total) * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
        if (route_scores_dev != nullptr) {
            // For decode (rows==1), read scores for resident experts.
            const int E = num_experts_;
            cold_scores_host.resize(static_cast<size_t>(E));
            LFM_CUDA(cudaMemcpyAsync(cold_scores_host.data(), route_scores_dev,
                                     static_cast<size_t>(E) * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
        }
        LFM_CUDA(cudaStreamSynchronize(stream));
        return 0;  // 0 cold experts
    }

    // Read back the cold list and their scores.
    cold_host.resize(static_cast<size_t>(cold_count));
    LFM_CUDA(cudaMemcpyAsync(cold_host.data(), cold_list_dev_.data(),
                             static_cast<size_t>(cold_count) * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));

    if (route_scores_dev != nullptr) {
        cold_scores_host.resize(static_cast<size_t>(cold_count));
        // Read scores for just the cold experts from device.
        // We need to gather them: cold_list_dev_ has the expert indices,
        // and route_scores_dev has per-expert scores. Use a small kernel or
        // just read the full score array (E floats, small for E=32).
        const int E = num_experts_;
        cold_scores_host.resize(static_cast<size_t>(E));
        LFM_CUDA(cudaMemcpyAsync(cold_scores_host.data(), route_scores_dev,
                                 static_cast<size_t>(E) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    }

    LFM_CUDA(cudaStreamSynchronize(stream));
    return cold_count;
}

void ExpertLayerCache::publish_pointer(int expert,
                                       const __nv_bfloat16* gate_up,
                                       const __nv_bfloat16* down,
                                       cudaStream_t stream) {
    LFM_CUDA(cudaMemcpyAsync(gate_up_ptrs_dev_.data() + expert, &gate_up,
                             sizeof(const __nv_bfloat16*),
                             cudaMemcpyHostToDevice, stream));
    LFM_CUDA(cudaMemcpyAsync(down_ptrs_dev_.data() + expert, &down,
                             sizeof(const __nv_bfloat16*),
                             cudaMemcpyHostToDevice, stream));
}

void ExpertLayerCache::promote(int expert, int slot, cudaStream_t stream,
                                float score) {
    if (expert < 0 || expert >= num_experts_) {
        throw std::invalid_argument("promote: expert out of range");
    }
    if (slot < 0 || slot >= capacity_) {
        throw std::invalid_argument("promote: slot out of range");
    }
    if (gate_up_host_dev_[static_cast<size_t>(expert)] == nullptr) {
        throw std::runtime_error("promote: host source not set for expert");
    }
    Slot& s = slots_[static_cast<size_t>(slot)];
    // Evict the current occupant back to its host source.
    if (s.expert >= 0 && s.expert != expert) {
        expert_slot_[static_cast<size_t>(s.expert)] = -1;
        publish_pointer(s.expert, gate_up_host_dev_[static_cast<size_t>(s.expert)],
                        down_host_dev_[static_cast<size_t>(s.expert)], stream);
    }
    // Copy weights host -> cache slot.
    LFM_CUDA(cudaMemcpyAsync(s.gate_up.data(),
                             gate_up_host_dev_[static_cast<size_t>(expert)],
                             gate_up_bytes_, cudaMemcpyDefault, stream));
    LFM_CUDA(cudaMemcpyAsync(s.down.data(),
                             down_host_dev_[static_cast<size_t>(expert)],
                             down_bytes_, cudaMemcpyDefault, stream));
    s.expert = expert;
    expert_slot_[static_cast<size_t>(expert)] = slot;
    last_used_[static_cast<size_t>(expert)] = ++lru_tick_;
    last_score_[static_cast<size_t>(expert)] = score;
    // Point the table at the slot only after the copies are ordered on-stream.
    publish_pointer(expert, s.gate_up.data(), s.down.data(), stream);
    // Keep device mirror in sync for the next resolve_on_device call.
    sync_expert_slot_to_device();
}

void ExpertLayerCache::evict(int slot, cudaStream_t stream) {
    if (slot < 0 || slot >= capacity_) {
        throw std::invalid_argument("evict: slot out of range");
    }
    Slot& s = slots_[static_cast<size_t>(slot)];
    if (s.expert < 0) return;
    const int expert = s.expert;
    expert_slot_[static_cast<size_t>(expert)] = -1;
    s.expert = -1;
    publish_pointer(expert, gate_up_host_dev_[static_cast<size_t>(expert)],
                    down_host_dev_[static_cast<size_t>(expert)], stream);
    sync_expert_slot_to_device();
}

int ExpertLayerCache::choose_victim() const {
    if (capacity_ <= 0) return -1;
    // Prefer an empty slot; otherwise evict per the cache policy:
    //  * Lru   -> least-recently-used (default; best measured on LFM2-8B-A1B).
    //  * Score -> lowest routing-likelihood (router-guided residency).
    // Ties break to the lowest slot index for determinism.
    int best = -1;
    auto better = [this](int a_slot, int b_slot) {
        const Slot& a = slots_[static_cast<size_t>(a_slot)];
        const Slot& b = slots_[static_cast<size_t>(b_slot)];
        if (policy_ == ExpertCachePolicy::Score) {
            const float sa = last_score_[static_cast<size_t>(a.expert)];
            const float sb = last_score_[static_cast<size_t>(b.expert)];
            return sa < sb;
        }
        const uint64_t ua = last_used_[static_cast<size_t>(a.expert)];
        const uint64_t ub = last_used_[static_cast<size_t>(b.expert)];
        return ua < ub;
    };
    for (int i = 0; i < capacity_; ++i) {
        const Slot& s = slots_[static_cast<size_t>(i)];
        if (s.expert < 0) return i;
        if (best < 0 || better(i, best)) best = i;
    }
    return best;
}

void ExpertLayerCache::touch(int expert) {
    if (expert < 0 || expert >= num_experts_) return;
    const int slot = expert_slot_[static_cast<size_t>(expert)];
    if (slot < 0) return;  // host-resident experts have no cache slot
    last_used_[static_cast<size_t>(expert)] = ++lru_tick_;
}

void ExpertLayerCache::touch(int expert, float score) {
    if (expert < 0 || expert >= num_experts_) return;
    const int slot = expert_slot_[static_cast<size_t>(expert)];
    if (slot < 0) return;  // host-resident experts have no cache slot
    last_used_[static_cast<size_t>(expert)] = ++lru_tick_;
    last_score_[static_cast<size_t>(expert)] = score;
}

bool ExpertLayerCache::ensure_resident(int expert, cudaStream_t stream,
                                        float score) {
    if (expert < 0 || expert >= num_experts_) {
        throw std::invalid_argument("ensure_resident: expert out of range");
    }
    if (expert_slot_[static_cast<size_t>(expert)] >= 0) {
        touch(expert, score);  // already cached: refresh recency + score
        return false;
    }
    const int slot = choose_victim();
    if (slot < 0) return false;  // capacity 0 -> everything stays host-resident
    promote(expert, slot, stream, score);
    return true;
}

int ExpertLayerCache::prefetch(int n, cudaStream_t stream) {
    if (n <= 0 || capacity_ <= 0) return 0;
    // Rank not-yet-resident experts by recency (most-recently-used first), so
    // the ones most likely to be routed next are pulled onto the GPU.
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(num_experts_));
    for (int e = 0; e < num_experts_; ++e) {
        if (expert_slot_[static_cast<size_t>(e)] < 0) {
            candidates.push_back(e);
        }
    }
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + std::min<int>(n, static_cast<int>(candidates.size())),
        candidates.end(),
        [this](int a, int b) {
            return last_used_[static_cast<size_t>(a)] >
                   last_used_[static_cast<size_t>(b)];
        });
    int promoted = 0;
    const int limit = std::min<int>(n, static_cast<int>(candidates.size()));
    for (int i = 0; i < limit; ++i) {
        const int e = candidates[static_cast<size_t>(i)];
        // Skip if it got resident via another path while we were ranking.
        if (expert_slot_[static_cast<size_t>(e)] >= 0) continue;
        const int slot = choose_victim();
        if (slot < 0) break;
        promote(e, slot, stream);
        ++promoted;
    }
    return promoted;
}

int ExpertLayerCache::prefetch_list(const std::vector<int>& ranked,
                                     const std::vector<float>& scores, int n,
                                     cudaStream_t stream) {
    if (n <= 0 || capacity_ <= 0) return 0;
    if (ranked.size() != scores.size()) {
        throw std::invalid_argument("prefetch_list: ranked/scores size mismatch");
    }
    int promoted = 0;
    const int limit = std::min<int>(n, static_cast<int>(ranked.size()));
    for (int i = 0; i < limit; ++i) {
        const int e = ranked[static_cast<size_t>(i)];
        if (e < 0 || e >= num_experts_) continue;
        // Already resident: refresh its score so it stays where it is.
        if (expert_slot_[static_cast<size_t>(e)] >= 0) {
            last_score_[static_cast<size_t>(e)] = scores[static_cast<size_t>(i)];
            continue;
        }
        const int slot = choose_victim();
        if (slot < 0) break;
        // Anti-thrash: never evict a resident expert whose score is >= this
        // candidate's, since `ranked` is descending in likelihood any further
        // candidate is also <=. Stop prefetching for this step.
        const Slot& victim = slots_[static_cast<size_t>(slot)];
        if (victim.expert >= 0) {
            const float victim_score =
                last_score_[static_cast<size_t>(victim.expert)];
            const float cand_score = scores[static_cast<size_t>(i)];
            if (cand_score <= victim_score) break;
        }
        promote(e, slot, stream, scores[static_cast<size_t>(i)]);
        ++promoted;
    }
    return promoted;
}

void ExpertLayerCache::seed(const std::vector<int>& experts,
                            cudaStream_t stream) {
    const int count = std::min<int>(static_cast<int>(experts.size()), capacity_);
    for (int i = 0; i < count; ++i) {
        promote(experts[static_cast<size_t>(i)], i, stream);
    }
}

} // namespace lfm
