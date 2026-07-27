#pragma once

#include "lfm/checkpoint/formats/safetensors.hpp"

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

namespace lfm {

class PinnedExpertCache;

class ExpertHostLease {
public:
    ExpertHostLease() = default;
    ExpertHostLease(PinnedExpertCache* cache, int slot_idx);
    ~ExpertHostLease();

    ExpertHostLease(const ExpertHostLease&) = delete;
    ExpertHostLease& operator=(const ExpertHostLease&) = delete;
    ExpertHostLease(ExpertHostLease&& other) noexcept;
    ExpertHostLease& operator=(ExpertHostLease&& other) noexcept;

    const std::byte* gate_up() const;
    const std::byte* down() const;
    int slot_index() const { return slot_idx_; }
    int layer() const { return layer_; }
    int expert() const { return expert_; }
    bool valid() const { return cache_ != nullptr; }
    void release();

private:
    PinnedExpertCache* cache_ = nullptr;
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

class PinnedExpertCache {
public:
    PinnedExpertCache(std::size_t budget_bytes, std::size_t bytes_per_expert,
                      std::size_t gate_up_bytes, std::size_t down_bytes);
    ~PinnedExpertCache();

    PinnedExpertCache(const PinnedExpertCache&) = delete;
    PinnedExpertCache& operator=(const PinnedExpertCache&) = delete;

    using LoaderFn = std::function<void(std::span<std::byte> gate_up_dest, std::span<std::byte> down_dest)>;

    // Acquires a lease. If the expert is missing, invokes loader_fn while other
    // concurrent waiters coalesce and wait on the same load.
    ExpertHostLease acquire(int layer, int expert, const LoaderFn& loader_fn);

    void release_slot(int slot_idx);

    std::size_t hits() const { return hits_; }
    std::size_t misses() const { return misses_; }
    std::size_t evictions() const { return evictions_; }
    std::size_t bytes_read() const { return bytes_read_; }
    double total_wait_time_ms() const { return total_wait_time_ms_; }

private:
    friend class ExpertHostLease;

    std::mutex mutex_;
    std::size_t bytes_per_expert_ = 0;
    std::size_t gate_up_bytes_ = 0;
    std::size_t down_bytes_ = 0;
    int capacity_ = 0;

    void* arena_ = nullptr;

    struct CacheSlot {
        int layer = -1;
        int expert = -1;
        std::byte* gate_up_ptr = nullptr;
        std::byte* down_ptr = nullptr;
        int ref_count = 0;
        uint64_t last_used = 0;
        bool loading = false;
        uint64_t generation = 0;
        std::vector<std::shared_ptr<std::promise<void>>> waiters;
        std::shared_future<void> shared_fut;
    };

    std::vector<CacheSlot> slots_;
    uint64_t tick_ = 0;

    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
    std::size_t evictions_ = 0;
    std::size_t bytes_read_ = 0;
    double total_wait_time_ms_ = 0.0;

    int find_slot(int layer, int expert);
    int choose_victim_slot();
};

} // namespace lfm
