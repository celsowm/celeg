#include "celeg/backend/cuda/moe/expert_residency.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace celeg {

namespace {

constexpr uint64_t kLfuDecayInterval = 256;
constexpr float kLfuDecayFactor = 0.5f;

__global__ void resolve_residency_kernel(const int* sel_dev,
                                         const int* expert_slot_dev,
                                         int* cold_flags_dev,
                                         int* cold_list_dev,
                                         int* cold_count_dev,
                                         int* active_flags_dev,
                                         int* active_list_dev,
                                         int* active_count_dev,
                                         int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int e = sel_dev[idx];
    if (e < 0) return;
    if (atomicCAS(&active_flags_dev[e], 0, 1) == 0) {
        const int active_pos = atomicAdd(active_count_dev, 1);
        active_list_dev[active_pos] = e;
    }
    if (expert_slot_dev[e] < 0) {
        if (atomicCAS(&cold_flags_dev[e], 0, 1) == 0) {
            const int pos = atomicAdd(cold_count_dev, 1);
            cold_list_dev[pos] = e;
        }
    }
}

} // namespace

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

    if (capacity_ > 1) {
        const int probation = std::max(1, capacity_ / 4);
        protected_capacity_ = capacity_ - probation;
    }

    slots_.resize(static_cast<size_t>(capacity));
    for (Slot& slot : slots_) {
        slot.gate_up.reset(gate_up_bytes / sizeof(__nv_bfloat16));
        slot.down.reset(down_bytes / sizeof(__nv_bfloat16));
    }
    expert_slot_.assign(static_cast<size_t>(num_experts), -1);
    last_used_.assign(static_cast<size_t>(num_experts), 0);
    last_score_.assign(static_cast<size_t>(num_experts), kUnseen);
    gate_up_host_dev_.assign(static_cast<size_t>(num_experts), nullptr);
    down_host_dev_.assign(static_cast<size_t>(num_experts), nullptr);
    gate_up_ptrs_dev_.reset(static_cast<size_t>(num_experts));
    down_ptrs_dev_.reset(static_cast<size_t>(num_experts));

    expert_slot_dev_.reset(static_cast<size_t>(num_experts));
    CELEG_CUDA(cudaMemcpy(expert_slot_dev_.data(), expert_slot_.data(),
                         static_cast<size_t>(num_experts) * sizeof(int),
                         cudaMemcpyHostToDevice));
    cold_flags_dev_.reset(static_cast<size_t>(num_experts));
    cold_list_dev_.reset(static_cast<size_t>(num_experts));
    cold_count_dev_.reset(1);
    active_flags_dev_.reset(static_cast<size_t>(num_experts));
    active_list_dev_.reset(static_cast<size_t>(num_experts));
    active_count_dev_.reset(1);
}

void ExpertLayerCache::set_policy(ExpertCachePolicy policy) {
    if (policy_ == policy) return;
    policy_ = policy;
    protected_count_ = 0;
    for (Slot& slot : slots_) {
        slot.protected_entry = false;
        slot.batch_pinned = false;
        slot.observed_accesses = slot.expert >= 0 ? 1U : 0U;
    }
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
    requires_full_residency_ =
        std::all_of(gate_up_host_dev_.begin(), gate_up_host_dev_.end(),
                    [](const __nv_bfloat16* pointer) { return pointer == nullptr; }) &&
        std::all_of(down_host_dev_.begin(), down_host_dev_.end(),
                    [](const __nv_bfloat16* pointer) { return pointer == nullptr; });
    gate_up_ptrs_host_ = gate_up_host_dev_;
    down_ptrs_host_ = down_host_dev_;
    CELEG_CUDA(cudaMemcpy(gate_up_ptrs_dev_.data(), gate_up_host_dev_.data(),
                         static_cast<size_t>(num_experts_) * sizeof(const __nv_bfloat16*),
                         cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(down_ptrs_dev_.data(), down_host_dev_.data(),
                         static_cast<size_t>(num_experts_) * sizeof(const __nv_bfloat16*),
                         cudaMemcpyHostToDevice));
    sync_expert_slot_to_device();
}

void ExpertLayerCache::sync_expert_slot_to_device(cudaStream_t stream) {
    if (stream != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(expert_slot_dev_.data(), expert_slot_.data(),
                                  static_cast<size_t>(num_experts_) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));
    } else {
        CELEG_CUDA(cudaMemcpy(expert_slot_dev_.data(), expert_slot_.data(),
                             static_cast<size_t>(num_experts_) * sizeof(int),
                             cudaMemcpyHostToDevice));
    }
}

int ExpertLayerCache::resolve_on_device(
    const int* sel_dev, const float* route_scores_dev,
    int rows, int K, cudaStream_t stream,
    std::vector<int>& cold_host,
    std::vector<float>& cold_scores_host,
    std::vector<int>& active_host) {
    const int total = rows * K;
    if (total == 0) return 0;

    // The caller waits for the previous FFN and prefetch events before this
    // method, so slots pinned for the preceding batch are safe to release now.
    for (Slot& slot : slots_) slot.batch_pinned = false;

    const int zero = 0;
    CELEG_CUDA(cudaMemcpyAsync(cold_count_dev_.data(), &zero, sizeof(int),
                              cudaMemcpyHostToDevice, stream));
    CELEG_CUDA(cudaMemsetAsync(cold_flags_dev_.data(), 0,
                              static_cast<size_t>(num_experts_) * sizeof(int), stream));
    CELEG_CUDA(cudaMemsetAsync(active_flags_dev_.data(), 0,
                              static_cast<size_t>(num_experts_) * sizeof(int), stream));
    CELEG_CUDA(cudaMemsetAsync(active_count_dev_.data(), 0, sizeof(int), stream));

    const int block = 256;
    const int grid = (total + block - 1) / block;
    resolve_residency_kernel<<<grid, block, 0, stream>>>(
        sel_dev, expert_slot_dev_.data(), cold_flags_dev_.data(),
        cold_list_dev_.data(), cold_count_dev_.data(), active_flags_dev_.data(),
        active_list_dev_.data(), active_count_dev_.data(), total);

    int counts[2] = {0, 0};
    CELEG_CUDA(cudaMemcpyAsync(&counts[0], cold_count_dev_.data(), sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    CELEG_CUDA(cudaMemcpyAsync(&counts[1], active_count_dev_.data(), sizeof(int),
                              cudaMemcpyDeviceToHost, stream));
    CELEG_CUDA(cudaEventRecord(discovery_done_event_.get(), stream));
    cudaError_t discovery_status = cudaEventQuery(discovery_done_event_.get());
    if (discovery_status == cudaErrorNotReady) {
        CELEG_CUDA(cudaEventSynchronize(discovery_done_event_.get()));
    } else {
        CELEG_CUDA(discovery_status);
    }
    const int cold_count = counts[0];
    const int active_count = counts[1];

    active_host.resize(static_cast<size_t>(active_count));
    if (active_count > 0) {
        CELEG_CUDA(cudaMemcpy(active_host.data(), active_list_dev_.data(),
                              static_cast<size_t>(active_count) * sizeof(int),
                              cudaMemcpyDeviceToHost));
    }

    if (cold_count == 0) return 0;
    if (!reserve_probation_slots(cold_count) && requires_full_residency_) {
        throw std::runtime_error(
            "disk-backed MoE routed more unique cold experts than the GPU cache can hold; "
            "increase --expert-cache-per-layer or reduce the prefill chunk");
    }

    cold_host.resize(static_cast<size_t>(cold_count));
    CELEG_CUDA(cudaMemcpy(cold_host.data(), cold_list_dev_.data(),
                          static_cast<size_t>(cold_count) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (route_scores_dev != nullptr) {
        cold_scores_host.resize(static_cast<size_t>(num_experts_));
        CELEG_CUDA(cudaMemcpy(cold_scores_host.data(), route_scores_dev,
                                  static_cast<size_t>(num_experts_) * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    }
    return cold_count;
}

void ExpertLayerCache::publish_pointer(int expert,
                                       const __nv_bfloat16* gate_up,
                                       const __nv_bfloat16* down,
                                       cudaStream_t stream) {
    (void)stream;
    gate_up_ptrs_host_[static_cast<size_t>(expert)] = gate_up;
    down_ptrs_host_[static_cast<size_t>(expert)] = down;
}

void ExpertLayerCache::publish_pointer_immediate(int expert,
                                                 const __nv_bfloat16* gate_up,
                                                 const __nv_bfloat16* down) {
    gate_up_ptrs_host_[static_cast<size_t>(expert)] = gate_up;
    down_ptrs_host_[static_cast<size_t>(expert)] = down;
    const size_t offset = static_cast<size_t>(expert) * sizeof(const __nv_bfloat16*);
    CELEG_CUDA(cudaMemcpy(reinterpret_cast<char*>(gate_up_ptrs_dev_.data()) + offset,
                         &gate_up, sizeof(gate_up), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(reinterpret_cast<char*>(down_ptrs_dev_.data()) + offset,
                         &down, sizeof(down), cudaMemcpyHostToDevice));
}

void ExpertLayerCache::sync_residency_tables(cudaStream_t stream) {
    if (gate_up_ptrs_host_.size() != static_cast<size_t>(num_experts_) ||
        down_ptrs_host_.size() != static_cast<size_t>(num_experts_)) {
        throw std::runtime_error("residency pointer table is not initialized");
    }
    const size_t bytes = static_cast<size_t>(num_experts_) * sizeof(const __nv_bfloat16*);
    if (stream != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(gate_up_ptrs_dev_.data(), gate_up_ptrs_host_.data(),
                                  bytes, cudaMemcpyHostToDevice, stream));
        CELEG_CUDA(cudaMemcpyAsync(down_ptrs_dev_.data(), down_ptrs_host_.data(),
                                  bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CELEG_CUDA(cudaMemcpy(gate_up_ptrs_dev_.data(), gate_up_ptrs_host_.data(),
                             bytes, cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(down_ptrs_dev_.data(), down_ptrs_host_.data(),
                             bytes, cudaMemcpyHostToDevice));
    }
    sync_expert_slot_to_device(stream);
}

void ExpertLayerCache::mark_probation(int slot) {
    Slot& entry = slots_.at(static_cast<size_t>(slot));
    if (entry.protected_entry) {
        entry.protected_entry = false;
        --protected_count_;
    }
    entry.observed_accesses = entry.expert >= 0 ? 1U : 0U;
}

int ExpertLayerCache::probation_entries() const {
    int count = 0;
    for (const Slot& entry : slots_) {
        if (!entry.protected_entry) ++count;
    }
    return count;
}

bool ExpertLayerCache::reserve_probation_slots(int required) {
    if (required <= 0) return true;
    if (required > capacity_) return false;
    while (probation_entries() < required) {
        const int victim = choose_coldest_protected();
        if (victim < 0) break;
        mark_probation(victim);
    }
    return probation_entries() >= required;
}

double ExpertLayerCache::priority(int expert) const {
    const float heat = std::max(0.0f, last_score_[static_cast<size_t>(expert)]);
    const uint64_t age = lru_tick_ - last_used_[static_cast<size_t>(expert)];
    return static_cast<double>(heat) +
           1.0 / (1.0 + static_cast<double>(age));
}

int ExpertLayerCache::choose_coldest_protected(int except_slot) const {
    int best = -1;
    for (int slot = 0; slot < capacity_; ++slot) {
        if (slot == except_slot) continue;
        const Slot& entry = slots_[static_cast<size_t>(slot)];
        if (entry.expert < 0 || !entry.protected_entry || entry.batch_pinned) continue;
        if (best < 0 || priority(entry.expert) <
                            priority(slots_[static_cast<size_t>(best)].expert) ||
            (priority(entry.expert) ==
                 priority(slots_[static_cast<size_t>(best)].expert) &&
             entry.last_used < slots_[static_cast<size_t>(best)].last_used)) {
            best = slot;
        }
    }
    return best;
}

void ExpertLayerCache::promote_to_protected(int slot) {
    if (policy_ != ExpertCachePolicy::LayerLocalLfuLru ||
        protected_capacity_ <= 0) {
        return;
    }
    Slot& entry = slots_.at(static_cast<size_t>(slot));
    if (entry.expert < 0 || entry.protected_entry) return;

    if (protected_count_ >= protected_capacity_) {
        const int victim = choose_coldest_protected(slot);
        if (victim >= 0) mark_probation(victim);
    }
    if (protected_count_ < protected_capacity_) {
        entry.protected_entry = true;
        ++protected_count_;
    }
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

    Slot& entry = slots_[static_cast<size_t>(slot)];
    if (entry.protected_entry) --protected_count_;
    if (entry.expert >= 0 && entry.expert != expert) {
        expert_slot_[static_cast<size_t>(entry.expert)] = -1;
        publish_pointer(entry.expert,
                        gate_up_host_dev_[static_cast<size_t>(entry.expert)],
                        down_host_dev_[static_cast<size_t>(entry.expert)], stream);
    }

    CELEG_CUDA(cudaMemcpyAsync(entry.gate_up.data(),
                              gate_up_host_dev_[static_cast<size_t>(expert)],
                              gate_up_bytes_, cudaMemcpyDefault, stream));
    CELEG_CUDA(cudaMemcpyAsync(entry.down.data(),
                              down_host_dev_[static_cast<size_t>(expert)],
                              down_bytes_, cudaMemcpyDefault, stream));
    entry.expert = expert;
    entry.protected_entry = false;
    entry.batch_pinned = false;
    entry.observed_accesses = 0;
    entry.last_used = ++lru_tick_;
    expert_slot_[static_cast<size_t>(expert)] = slot;
    last_used_[static_cast<size_t>(expert)] = entry.last_used;
    if (policy_ != ExpertCachePolicy::LayerLocalLfuLru) {
        last_score_[static_cast<size_t>(expert)] = score;
    }
    publish_pointer(expert, entry.gate_up.data(), entry.down.data(), stream);
}

void ExpertLayerCache::promote(int expert, int slot,
                               const __nv_bfloat16* gate_up_src,
                               const __nv_bfloat16* down_src,
                               cudaStream_t stream, float score) {
    if (expert < 0 || expert >= num_experts_) {
        throw std::invalid_argument("promote: expert out of range");
    }
    if (slot < 0 || slot >= capacity_) {
        throw std::invalid_argument("promote: slot out of range");
    }
    if (gate_up_src == nullptr || down_src == nullptr) {
        throw std::invalid_argument("promote: source pointer is null");
    }

    Slot& entry = slots_[static_cast<size_t>(slot)];
    if (entry.protected_entry) --protected_count_;
    if (entry.expert >= 0 && entry.expert != expert) {
        expert_slot_[static_cast<size_t>(entry.expert)] = -1;
        publish_pointer(entry.expert,
                        gate_up_host_dev_[static_cast<size_t>(entry.expert)],
                        down_host_dev_[static_cast<size_t>(entry.expert)], stream);
    }

    // This overload is used by the disk-backed residency path.  Its sources
    // are ordinary/pinned host buffers owned by HostExpertCache, not CUDA
    // device pointers.  Keep the direction explicit: cudaMemcpyDefault can
    // misclassify a large Windows pinned allocation as a device/UVA pointer.
    CELEG_CUDA(cudaMemcpyAsync(entry.gate_up.data(), gate_up_src,
                              gate_up_bytes_, cudaMemcpyHostToDevice, stream));
    CELEG_CUDA(cudaMemcpyAsync(entry.down.data(), down_src,
                              down_bytes_, cudaMemcpyHostToDevice, stream));
    entry.expert = expert;
    entry.protected_entry = false;
    entry.batch_pinned = false;
    entry.observed_accesses = 0;
    entry.last_used = ++lru_tick_;
    expert_slot_[static_cast<size_t>(expert)] = slot;
    last_used_[static_cast<size_t>(expert)] = entry.last_used;
    if (policy_ != ExpertCachePolicy::LayerLocalLfuLru) {
        last_score_[static_cast<size_t>(expert)] = score;
    }
    publish_pointer(expert, entry.gate_up.data(), entry.down.data(), stream);
}

void ExpertLayerCache::evict(int slot, cudaStream_t stream) {
    if (slot < 0 || slot >= capacity_) {
        throw std::invalid_argument("evict: slot out of range");
    }
    Slot& entry = slots_[static_cast<size_t>(slot)];
    if (entry.expert < 0) return;
    const int expert = entry.expert;
    if (entry.protected_entry) --protected_count_;
    expert_slot_[static_cast<size_t>(expert)] = -1;
    entry.expert = -1;
    entry.protected_entry = false;
    entry.batch_pinned = false;
    entry.observed_accesses = 0;
    entry.last_used = 0;
    publish_pointer(expert, gate_up_host_dev_[static_cast<size_t>(expert)],
                    down_host_dev_[static_cast<size_t>(expert)], stream);
}

int ExpertLayerCache::choose_victim(bool speculative) const {
    if (capacity_ <= 0) return -1;
    for (int slot = 0; slot < capacity_; ++slot) {
        const Slot& entry = slots_[static_cast<size_t>(slot)];
        if (entry.expert < 0 && !entry.batch_pinned) return slot;
    }

    auto better = [this](int candidate, int current) {
        const Slot& a = slots_[static_cast<size_t>(candidate)];
        const Slot& b = slots_[static_cast<size_t>(current)];
        if (policy_ == ExpertCachePolicy::Score) {
            const float a_score = last_score_[static_cast<size_t>(a.expert)];
            const float b_score = last_score_[static_cast<size_t>(b.expert)];
            if (a_score != b_score) return a_score < b_score;
        } else if (policy_ == ExpertCachePolicy::LayerLocalLfuLru) {
            const double a_priority = priority(a.expert);
            const double b_priority = priority(b.expert);
            if (a_priority != b_priority) return a_priority < b_priority;
        }
        return a.last_used < b.last_used;
    };

    int best = -1;
    if (policy_ == ExpertCachePolicy::LayerLocalLfuLru) {
        for (int slot = 0; slot < capacity_; ++slot) {
            const Slot& entry = slots_[static_cast<size_t>(slot)];
            if (entry.protected_entry || entry.batch_pinned) continue;
            if (best < 0 || better(slot, best)) best = slot;
        }
        if (best >= 0 || speculative) return best;
    }

    for (int slot = 0; slot < capacity_; ++slot) {
        if (slots_[static_cast<size_t>(slot)].batch_pinned) continue;
        if (best < 0 || better(slot, best)) best = slot;
    }
    return best;
}

void ExpertLayerCache::touch(int expert) {
    touch(expert, kUnseen);
}

void ExpertLayerCache::touch(int expert, float score) {
    if (expert < 0 || expert >= num_experts_) return;
    const uint64_t tick = ++lru_tick_;
    if (policy_ == ExpertCachePolicy::LayerLocalLfuLru) {
        if (tick % kLfuDecayInterval == 0) {
            for (float& heat : last_score_) {
                if (heat > 0.0f) heat *= kLfuDecayFactor;
            }
        }
        float& heat = last_score_[static_cast<size_t>(expert)];
        if (heat == kUnseen || heat < 0.0f) heat = 0.0f;
        heat += 1.0f;
    }

    const int slot = expert_slot_[static_cast<size_t>(expert)];
    if (slot < 0) return;
    Slot& entry = slots_[static_cast<size_t>(slot)];
    entry.batch_pinned = true;
    entry.last_used = tick;
    last_used_[static_cast<size_t>(expert)] = tick;
    if (entry.observed_accesses < std::numeric_limits<uint32_t>::max()) {
        ++entry.observed_accesses;
    }
    if (policy_ != ExpertCachePolicy::LayerLocalLfuLru) {
        last_score_[static_cast<size_t>(expert)] = score;
    } else if (entry.observed_accesses >= 2) {
        promote_to_protected(slot);
    }
}

void ExpertLayerCache::pin_active(int expert, float score) {
    touch(expert, score);
}

bool ExpertLayerCache::ensure_resident(int expert, cudaStream_t stream,
                                       float score) {
    if (expert < 0 || expert >= num_experts_) {
        throw std::invalid_argument("ensure_resident: expert out of range");
    }
    if (resident(expert)) {
        return false;
    }
    const int slot = choose_victim(false);
    if (slot < 0) return false;
    promote(expert, slot, stream, score);
    return true;
}

bool ExpertLayerCache::ensure_resident(int expert,
                                       const __nv_bfloat16* gate_up_src,
                                       const __nv_bfloat16* down_src,
                                       cudaStream_t stream, float score) {
    if (expert < 0 || expert >= num_experts_) {
        throw std::invalid_argument("ensure_resident: expert out of range");
    }
    if (resident(expert)) {
        return false;
    }
    const int slot = choose_victim(false);
    if (slot < 0) return false;
    promote(expert, slot, gate_up_src, down_src, stream, score);
    return true;
}

int ExpertLayerCache::prefetch(int n, cudaStream_t stream) {
    if (n <= 0 || capacity_ <= 0) return 0;
    std::vector<int> candidates;
    candidates.reserve(static_cast<size_t>(num_experts_));
    for (int expert = 0; expert < num_experts_; ++expert) {
        if (!resident(expert)) candidates.push_back(expert);
    }
    const int limit = std::min<int>(n, static_cast<int>(candidates.size()));
    std::partial_sort(
        candidates.begin(), candidates.begin() + limit, candidates.end(),
        [this](int a, int b) {
            return last_used_[static_cast<size_t>(a)] >
                   last_used_[static_cast<size_t>(b)];
        });

    int promoted = 0;
    for (int index = 0; index < limit; ++index) {
        const int expert = candidates[static_cast<size_t>(index)];
        if (resident(expert)) continue;
        const int slot = choose_victim(true);
        if (slot < 0) break;
        promote(expert, slot, stream);
        ++promoted;
    }
    if (promoted > 0) sync_residency_tables(stream);
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
    for (int index = 0; index < limit; ++index) {
        const int expert = ranked[static_cast<size_t>(index)];
        if (expert < 0 || expert >= num_experts_) continue;
        if (resident(expert)) {
            if (policy_ != ExpertCachePolicy::LayerLocalLfuLru) {
                last_score_[static_cast<size_t>(expert)] =
                    scores[static_cast<size_t>(index)];
            }
            continue;
        }

        const int slot = choose_victim(true);
        if (slot < 0) break;
        const Slot& victim = slots_[static_cast<size_t>(slot)];
        if (victim.expert >= 0 && policy_ != ExpertCachePolicy::LayerLocalLfuLru) {
            const float victim_score =
                last_score_[static_cast<size_t>(victim.expert)];
            const float candidate_score = scores[static_cast<size_t>(index)];
            if (candidate_score <= victim_score) break;
        }
        promote(expert, slot, stream, scores[static_cast<size_t>(index)]);
        ++promoted;
    }
    if (promoted > 0) sync_residency_tables(stream);
    return promoted;
}

void ExpertLayerCache::seed(const std::vector<int>& experts,
                            cudaStream_t stream, bool protect) {
    const int count = std::min<int>(static_cast<int>(experts.size()), capacity_);
    for (int slot = 0; slot < count; ++slot) {
        const int expert = experts[static_cast<size_t>(slot)];
        promote(expert, slot, stream);
        if (protect) promote_to_protected(slot);
    }
    if (count > 0) sync_residency_tables(stream);
}

const __nv_bfloat16* ExpertLayerCache::expert_gate_up_dev(int expert) const {
    const int slot = expert_slot(expert);
    return slot < 0 ? nullptr : slots_[static_cast<size_t>(slot)].gate_up.data();
}

const __nv_bfloat16* ExpertLayerCache::expert_down_dev(int expert) const {
    const int slot = expert_slot(expert);
    return slot < 0 ? nullptr : slots_[static_cast<size_t>(slot)].down.data();
}

} // namespace celeg
