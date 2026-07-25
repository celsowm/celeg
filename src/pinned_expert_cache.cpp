#include "lfm/pinned_expert_cache.hpp"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define HAVE_CUDA 1
#else
#define HAVE_CUDA 0
#endif

#include <cstring>
#include <cstdlib>
#include <limits>
#include <algorithm>

namespace lfm {

ExpertHostLease::ExpertHostLease(PinnedExpertCache* cache, int slot_idx)
    : cache_(cache), slot_idx_(slot_idx) {
    if (cache_ && slot_idx_ >= 0) {
        layer_ = cache_->slots_[slot_idx_].layer;
        expert_ = cache_->slots_[slot_idx_].expert;
    }
}

ExpertHostLease::~ExpertHostLease() {
    release();
}

ExpertHostLease::ExpertHostLease(ExpertHostLease&& other) noexcept
    : cache_(other.cache_),
      slot_idx_(other.slot_idx_),
      layer_(other.layer_),
      expert_(other.expert_) {
    other.cache_ = nullptr;
    other.slot_idx_ = -1;
    other.layer_ = -1;
    other.expert_ = -1;
}

ExpertHostLease& ExpertHostLease::operator=(ExpertHostLease&& other) noexcept {
    if (this != &other) {
        release();
        cache_ = other.cache_;
        slot_idx_ = other.slot_idx_;
        layer_ = other.layer_;
        expert_ = other.expert_;
        other.cache_ = nullptr;
        other.slot_idx_ = -1;
        other.layer_ = -1;
        other.expert_ = -1;
    }
    return *this;
}

const std::byte* ExpertHostLease::gate_up() const {
    if (!cache_ || slot_idx_ < 0) return nullptr;
    return cache_->slots_[slot_idx_].gate_up_ptr;
}

const std::byte* ExpertHostLease::down() const {
    if (!cache_ || slot_idx_ < 0) return nullptr;
    return cache_->slots_[slot_idx_].down_ptr;
}

void ExpertHostLease::release() {
    if (cache_ && slot_idx_ >= 0) {
        cache_->release_slot(slot_idx_);
        cache_ = nullptr;
        slot_idx_ = -1;
        layer_ = -1;
        expert_ = -1;
    }
}

// PinnedExpertCache implementation

PinnedExpertCache::PinnedExpertCache(std::size_t budget_bytes, std::size_t bytes_per_expert,
                                     std::size_t gate_up_bytes, std::size_t down_bytes)
    : bytes_per_expert_(bytes_per_expert),
      gate_up_bytes_(gate_up_bytes),
      down_bytes_(down_bytes) {
    if (bytes_per_expert == 0 || budget_bytes < bytes_per_expert) {
        // Fallback to at least 1 expert slot capacity
        capacity_ = 1;
    } else {
        capacity_ = static_cast<int>(budget_bytes / bytes_per_expert);
    }

    std::size_t arena_bytes = capacity_ * bytes_per_expert_;

#if HAVE_CUDA
    cudaError_t err = cudaMallocHost(&arena_, arena_bytes);
    if (err != cudaSuccess) {
        arena_ = std::malloc(arena_bytes);
    }
#else
    arena_ = std::malloc(arena_bytes);
#endif

    if (!arena_) {
        throw std::runtime_error("PinnedExpertCache: failed to allocate arena of size " +
                                 std::to_string(arena_bytes));
    }

    std::byte* base = static_cast<std::byte*>(arena_);
    slots_.resize(capacity_);
    for (int i = 0; i < capacity_; ++i) {
        slots_[i].gate_up_ptr = base + i * bytes_per_expert_;
        slots_[i].down_ptr = slots_[i].gate_up_ptr + gate_up_bytes_;
        slots_[i].layer = -1;
        slots_[i].expert = -1;
        slots_[i].ref_count = 0;
        slots_[i].last_used = 0;
        slots_[i].loading = false;
    }
}

PinnedExpertCache::~PinnedExpertCache() {
    if (arena_) {
#if HAVE_CUDA
        // Try cudaFreeHost, fallback to free if it wasn't registered/allocated by CUDA
        if (cudaFreeHost(arena_) != cudaSuccess) {
            std::free(arena_);
        }
#else
        std::free(arena_);
#endif
        arena_ = nullptr;
    }
}

int PinnedExpertCache::find_slot(int layer, int expert) {
    for (int i = 0; i < capacity_; ++i) {
        if (slots_[i].layer == layer && slots_[i].expert == expert) {
            return i;
        }
    }
    return -1;
}

int PinnedExpertCache::choose_victim_slot() {
    // 1. Prefer empty slots
    for (int i = 0; i < capacity_; ++i) {
        if (slots_[i].layer == -1 && slots_[i].ref_count == 0 && !slots_[i].loading) {
            return i;
        }
    }
    // 2. Choose LRU non-referenced, non-loading slot
    int victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < capacity_; ++i) {
        if (slots_[i].ref_count == 0 && !slots_[i].loading) {
            if (slots_[i].last_used < oldest) {
                oldest = slots_[i].last_used;
                victim = i;
            }
        }
    }
    return victim;
}

ExpertHostLease PinnedExpertCache::acquire(int layer, int expert, const LoaderFn& loader_fn) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int slot_idx = -1;

    {
        std::unique_lock<std::mutex> lock(mutex_);
        tick_++;
        slot_idx = find_slot(layer, expert);

        if (slot_idx >= 0) {
            // Cache hit
            slots_[slot_idx].last_used = tick_;
            if (slots_[slot_idx].loading) {
                // Someone else is loading it. Coalesce and wait.
                misses_++; // Still a miss from storage perspective but coalesced
                std::shared_ptr<std::promise<void>> p = std::make_shared<std::promise<void>>();
                slots_[slot_idx].waiters.push_back(p);
                lock.unlock();

                auto wait_start = std::chrono::high_resolution_clock::now();
                p->get_future().wait();
                auto wait_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> wait_elapsed = wait_end - wait_start;

                std::unique_lock<std::mutex> wait_lock(mutex_);
                total_wait_time_ms_ += wait_elapsed.count();
                slots_[slot_idx].ref_count++;
                return ExpertHostLease(this, slot_idx);
            } else {
                hits_++;
                slots_[slot_idx].ref_count++;
                return ExpertHostLease(this, slot_idx);
            }
        }

        // Cache miss: find a victim slot to evict
        misses_++;
        slot_idx = choose_victim_slot();
        if (slot_idx < 0) {
            throw std::runtime_error("PinnedExpertCache: out of free slots (all leased/loading)");
        }

        // Evict
        if (slots_[slot_idx].layer >= 0) {
            evictions_++;
        }

        slots_[slot_idx].layer = layer;
        slots_[slot_idx].expert = expert;
        slots_[slot_idx].last_used = tick_;
        slots_[slot_idx].ref_count = 1; // Mark as leased immediately
        slots_[slot_idx].loading = true;
    }

    // Perform disk load outside the global lock!
    try {
        std::span<std::byte> gate_up_dest(slots_[slot_idx].gate_up_ptr, gate_up_bytes_);
        std::span<std::byte> down_dest(slots_[slot_idx].down_ptr, down_bytes_);

        loader_fn(gate_up_dest, down_dest);
        bytes_read_ += gate_up_bytes_ + down_bytes_;
    } catch (...) {
        // Rollback state under lock
        std::lock_guard<std::mutex> lock(mutex_);
        slots_[slot_idx].layer = -1;
        slots_[slot_idx].expert = -1;
        slots_[slot_idx].ref_count = 0;
        slots_[slot_idx].loading = false;
        for (auto& waiter : slots_[slot_idx].waiters) {
            waiter->set_exception(std::current_exception());
        }
        slots_[slot_idx].waiters.clear();
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_[slot_idx].loading = false;
        for (auto& waiter : slots_[slot_idx].waiters) {
            waiter->set_value();
        }
        slots_[slot_idx].waiters.clear();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_elapsed = end_time - start_time;
    total_wait_time_ms_ += total_elapsed.count();

    return ExpertHostLease(this, slot_idx);
}

void PinnedExpertCache::release_slot(int slot_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot_idx >= 0 && slot_idx < capacity_) {
        slots_[slot_idx].ref_count--;
        if (slots_[slot_idx].ref_count < 0) {
            slots_[slot_idx].ref_count = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// ExpertIoManager Implementation
// ---------------------------------------------------------------------------

ExpertIoManager::ExpertIoManager(int num_workers, int queue_depth)
    : max_queue_depth_(queue_depth) {
    if (num_workers <= 0) num_workers = 1;
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::pair<std::function<void()>, std::shared_ptr<std::promise<void>>> item;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_not_empty_.wait(lock, [this]() {
                        return shutdown_ || !queue_.empty();
                    });
                    if (shutdown_ && queue_.empty()) {
                        return;
                    }
                    item = std::move(queue_.front());
                    queue_.pop();
                    cv_not_full_.notify_one();
                }
                try {
                    item.first();
                    item.second->set_value();
                } catch (...) {
                    item.second->set_exception(std::current_exception());
                }
            }
        });
    }
}

ExpertIoManager::~ExpertIoManager() {
    shutdown_ = true;
    cv_not_empty_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::future<void> ExpertIoManager::submit(std::function<void()> task) {
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() {
            return shutdown_ || static_cast<int>(queue_.size()) < max_queue_depth_;
        });
        if (shutdown_) {
            throw std::runtime_error("ExpertIoManager is shutting down");
        }
        queue_.emplace(std::move(task), promise);
        cv_not_empty_.notify_one();
    }
    return future;
}

} // namespace lfm
