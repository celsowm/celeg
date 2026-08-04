#include "celeg/runtime/cache/pinned_expert_cache.hpp"

#include <cstring>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <cmath>

namespace celeg {

namespace {
void* default_host_allocate(std::size_t bytes) { return std::malloc(bytes); }
void default_host_deallocate(void* pointer) { std::free(pointer); }
} // namespace

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
                                     std::size_t gate_up_bytes, std::size_t down_bytes,
                                     HostAllocateFn allocate, HostDeallocateFn deallocate)
    : bytes_per_expert_(bytes_per_expert),
      gate_up_bytes_(gate_up_bytes),
      down_bytes_(down_bytes),
      deallocate_(deallocate ? deallocate : default_host_deallocate) {
    if (bytes_per_expert == 0 || budget_bytes < bytes_per_expert) {
        // Fallback to at least 1 expert slot capacity
        capacity_ = 1;
    } else {
        capacity_ = static_cast<int>(budget_bytes / bytes_per_expert);
    }

    std::size_t arena_bytes = capacity_ * bytes_per_expert_;

    // Allocation policy is injected by the owning backend.  The neutral
    // default is ordinary host storage; CUDA can provide pinned storage
    // without leaking CUDA headers into this runtime translation unit.
    arena_ = (allocate ? allocate : default_host_allocate)(arena_bytes);

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
        deallocate_(arena_);
        arena_ = nullptr;
    }
}

uint64_t PinnedExpertCache::expert_key(int layer, int expert) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
           static_cast<uint32_t>(expert);
}

void PinnedExpertCache::decay_heat_if_needed() {
    if (tick_ - last_decay_tick_ < kHeatDecayInterval) return;

    const uint64_t intervals = (tick_ - last_decay_tick_) / kHeatDecayInterval;
    const double factor = std::pow(kHeatDecayFactor, static_cast<double>(intervals));
    for (auto it = heat_.begin(); it != heat_.end();) {
        it->second.heat *= factor;
        if (it->second.heat < 1.0e-6) {
            it = heat_.erase(it);
        } else {
            ++it;
        }
    }
    last_decay_tick_ += intervals * kHeatDecayInterval;
}

void PinnedExpertCache::record_access(int layer, int expert) {
    decay_heat_if_needed();
    HeatEntry& entry = heat_[expert_key(layer, expert)];
    entry.heat += 1.0;
    entry.last_touch = tick_;
}

double PinnedExpertCache::current_heat(int layer, int expert) const {
    const auto it = heat_.find(expert_key(layer, expert));
    return it == heat_.end() ? 0.0 : it->second.heat;
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
    // 1. Prefer empty slots.
    for (int i = 0; i < capacity_; ++i) {
        if (slots_[i].layer == -1 && slots_[i].ref_count == 0 && !slots_[i].loading) {
            return i;
        }
    }

    // 2. Evict the coldest non-referenced, non-loading slot. Heat is a
    // periodically decayed access frequency; the recency bonus breaks close
    // calls in favour of keeping a recently used expert.
    int victim = -1;
    double lowest_priority = std::numeric_limits<double>::infinity();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < capacity_; ++i) {
        const CacheSlot& slot = slots_[i];
        if (slot.ref_count != 0 || slot.loading) continue;

        const uint64_t age = tick_ >= slot.last_used ? tick_ - slot.last_used : 0;
        const double recency_bonus = 1.0 / (1.0 + static_cast<double>(age));
        const double priority = current_heat(slot.layer, slot.expert) + recency_bonus;
        if (priority < lowest_priority ||
            (priority == lowest_priority && slot.last_used < oldest)) {
            lowest_priority = priority;
            oldest = slot.last_used;
            victim = i;
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
        record_access(layer, expert);
        slot_idx = find_slot(layer, expert);

        if (slot_idx >= 0) {
            // Cache hit
            slots_[slot_idx].last_used = tick_;
            if (slots_[slot_idx].loading) {
                // Someone else is loading it. Coalesce and wait.
                misses_++; // Still a miss from storage perspective but coalesced
                std::shared_future<void> fut = slots_[slot_idx].shared_fut;
                uint64_t expected_gen = slots_[slot_idx].generation;

                // Reserve ref_count before releasing lock!
                slots_[slot_idx].ref_count++;
                lock.unlock();

                auto wait_start = std::chrono::high_resolution_clock::now();
                try {
                    fut.get(); // get() propagates exceptions!
                } catch (...) {
                    auto wait_end = std::chrono::high_resolution_clock::now();
                    std::unique_lock<std::mutex> wait_lock(mutex_);
                    total_wait_time_ms_ += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
                    if (slots_[slot_idx].ref_count > 0) {
                        slots_[slot_idx].ref_count--;
                    }
                    throw;
                }
                auto wait_end = std::chrono::high_resolution_clock::now();

                std::unique_lock<std::mutex> wait_lock(mutex_);
                total_wait_time_ms_ += std::chrono::duration<double, std::milli>(wait_end - wait_start).count();

                // Validate slot identity and generation upon waking up
                if (slots_[slot_idx].layer != layer || slots_[slot_idx].expert != expert || slots_[slot_idx].generation != expected_gen) {
                    if (slots_[slot_idx].ref_count > 0) {
                        slots_[slot_idx].ref_count--;
                    }
                    throw std::runtime_error("PinnedExpertCache: race condition detected, slot content changed during wait");
                }

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
        slots_[slot_idx].generation++;

        // Initialize the promise and shared_future
        auto p = std::make_shared<std::promise<void>>();
        slots_[slot_idx].shared_fut = p->get_future().share();
        slots_[slot_idx].waiters = {p}; // Keep track of the main promise so we can set it
    }

    // Perform disk load outside the global lock!
    try {
        std::span<std::byte> gate_up_dest(slots_[slot_idx].gate_up_ptr, gate_up_bytes_);
        std::span<std::byte> down_dest(slots_[slot_idx].down_ptr, down_bytes_);

        loader_fn(gate_up_dest, down_dest);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            bytes_read_ += gate_up_bytes_ + down_bytes_;
        }
    } catch (...) {
        // Rollback state under lock
        std::lock_guard<std::mutex> lock(mutex_);
        slots_[slot_idx].layer = -1;
        slots_[slot_idx].expert = -1;
        if (slots_[slot_idx].ref_count > 0) {
            slots_[slot_idx].ref_count--; // Decrement loader's ref count
        }
        slots_[slot_idx].loading = false;
        if (!slots_[slot_idx].waiters.empty()) {
            slots_[slot_idx].waiters[0]->set_exception(std::current_exception());
        }
        slots_[slot_idx].waiters.clear();
        slots_[slot_idx].shared_fut = {};
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_[slot_idx].loading = false;
        if (!slots_[slot_idx].waiters.empty()) {
            slots_[slot_idx].waiters[0]->set_value();
        }
        slots_[slot_idx].waiters.clear();
        slots_[slot_idx].shared_fut = {};
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_elapsed = end_time - start_time;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        total_wait_time_ms_ += total_elapsed.count();
    }

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

} // namespace celeg
