#pragma once

#include "celeg/runtime/cache/expert_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <mutex>
#include <future>
#include <functional>
#include <chrono>
#include <stdexcept>
#include <span>
#include <memory>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <unordered_map>

namespace celeg {

class HostExpertCache;

class ExpertHostLease {
public:
    ExpertHostLease() = default;
    ExpertHostLease(HostExpertCache* cache, int slot_idx);
    ~ExpertHostLease();

    ExpertHostLease(const ExpertHostLease&) = delete;
    ExpertHostLease& operator=(const ExpertHostLease&) = delete;
    ExpertHostLease(ExpertHostLease&& other) noexcept;
    ExpertHostLease& operator=(ExpertHostLease&& other) noexcept;

    const std::byte* payload() const;
    int slot_index() const { return slot_idx_; }
    int layer() const { return layer_; }
    int expert() const { return expert_; }
    bool valid() const { return cache_ != nullptr; }
    void release();

private:
    HostExpertCache* cache_ = nullptr;
    int slot_idx_ = -1;
    int layer_ = -1;
    int expert_ = -1;
};

class ExpertIoManager {
public:
    ExpertIoManager(int num_workers, int queue_depth);
    ~ExpertIoManager();

    std::future<void> submit(std::function<void()> task);

private:
    std::vector<std::thread> workers_;
    std::queue<std::pair<std::function<void()>, std::shared_ptr<std::promise<void>>>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    std::atomic<bool> shutdown_{false};
    int max_queue_depth_ = 0;
};

// Backend-neutral host staging storage. The allocator is injected so CUDA
// can provide pinned/mapped memory, while ordinary tests and CPU-linked
// builds use malloc. Payload interpretation stays in the source adapter.
class HostExpertCache {
public:
    using HostAllocateFn = void* (*)(std::size_t);
    using HostDeallocateFn = void (*)(void*);

    HostExpertCache(std::size_t budget_bytes, std::size_t bytes_per_expert,
                      HostAllocateFn allocate = nullptr,
                      HostDeallocateFn deallocate = nullptr);
    ~HostExpertCache();

    HostExpertCache(const HostExpertCache&) = delete;
    HostExpertCache& operator=(const HostExpertCache&) = delete;

    using LoaderFn = std::function<void(std::span<std::byte> payload)>;

    // Acquires a lease. If the expert is missing, invokes loader_fn while other
    // concurrent waiters coalesce and wait on the same load.
    ExpertHostLease acquire(int layer, int expert, const LoaderFn& loader_fn);

    void release_slot(int slot_idx);

    std::size_t hits() const {
        std::lock_guard lock(mutex_);
        return hits_;
    }
    std::size_t misses() const {
        std::lock_guard lock(mutex_);
        return misses_;
    }
    std::size_t coalesced_waits() const {
        std::lock_guard lock(mutex_);
        return coalesced_waits_;
    }
    std::size_t evictions() const {
        std::lock_guard lock(mutex_);
        return evictions_;
    }
    std::size_t bytes_read() const {
        std::lock_guard lock(mutex_);
        return bytes_read_;
    }
    double total_wait_time_ms() const {
        std::lock_guard lock(mutex_);
        return total_wait_time_ms_;
    }

private:
    friend class ExpertHostLease;

    mutable std::mutex mutex_;
    std::size_t bytes_per_expert_ = 0;
    int capacity_ = 0;

    void* arena_ = nullptr;
    HostDeallocateFn deallocate_ = nullptr;

    struct CacheSlot {
        int layer = -1;
        int expert = -1;
        std::byte* payload_ptr = nullptr;
        int ref_count = 0;
        uint64_t last_used = 0;
        bool loading = false;
        uint64_t generation = 0;
        std::vector<std::shared_ptr<std::promise<void>>> waiters;
        std::shared_future<void> shared_fut;
    };

    struct HeatEntry {
        double heat = 0.0;
        uint64_t last_touch = 0;
    };

    std::vector<CacheSlot> slots_;
    std::unordered_map<uint64_t, HeatEntry> heat_;
    uint64_t tick_ = 0;
    uint64_t last_decay_tick_ = 0;

    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
    std::size_t coalesced_waits_ = 0;
    std::size_t evictions_ = 0;
    std::size_t bytes_read_ = 0;
    double total_wait_time_ms_ = 0.0;

    static constexpr uint64_t kHeatDecayInterval = 256;
    static constexpr double kHeatDecayFactor = 0.5;

    static uint64_t expert_key(int layer, int expert);
    void decay_heat_if_needed();
    void record_access(int layer, int expert);
    double current_heat(int layer, int expert) const;
    int find_slot(int layer, int expert);
    int choose_victim_slot();
};

} // namespace celeg
