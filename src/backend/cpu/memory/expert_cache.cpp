#include "../detail/expert_cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace celeg {

CpuExpertCache::CpuExpertCache(std::size_t budget_bytes)
    : budget_bytes_(budget_bytes),
      protected_budget_bytes_(budget_bytes * 3 / 4) {}

std::uint64_t CpuExpertCache::key(int layer, int expert) {
    return ExpertKey{layer, expert}.packed();
}

void CpuExpertCache::decay_if_needed() {
    if (tick_ - last_decay_tick_ < kDecayInterval) return;
    const std::uint64_t intervals =
        (tick_ - last_decay_tick_) / kDecayInterval;
    const double factor = std::pow(kDecayFactor, static_cast<double>(intervals));
    for (auto& [entry_key, entry] : entries_) {
        (void)entry_key;
        entry.heat *= factor;
    }
    last_decay_tick_ += intervals * kDecayInterval;
}

double CpuExpertCache::priority(const Entry& entry) const {
    const std::uint64_t age = tick_ >= entry.last_used
        ? tick_ - entry.last_used : 0;
    return entry.heat + 1.0 / (1.0 + static_cast<double>(age));
}

std::uint64_t CpuExpertCache::choose_victim(
    bool protected_entries, std::uint64_t except_key) const {
    std::uint64_t victim = kNoKey;
    bool found = false;
    double lowest = std::numeric_limits<double>::infinity();
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (const auto& [entry_key, entry] : entries_) {
        if (entry_key == except_key || !entry.weights ||
            entry.protected_entry != protected_entries ||
            entry.weights.use_count() != 1) {
            continue;
        }
        const double candidate = priority(entry);
        if (!found || candidate < lowest ||
            (candidate == lowest && entry.last_used < oldest)) {
            found = true;
            victim = entry_key;
            lowest = candidate;
            oldest = entry.last_used;
        }
    }
    return found ? victim : kNoKey;
}

void CpuExpertCache::demote_coldest_protected(std::uint64_t except_key) {
    const std::uint64_t victim = choose_victim(true, except_key);
    if (victim == kNoKey) return;
    Entry& entry = entries_.at(victim);
    entry.protected_entry = false;
    entry.observed_accesses = 1;
    protected_bytes_ -= entry.bytes;
}

void CpuExpertCache::promote_to_protected(std::uint64_t entry_key) {
    Entry& entry = entries_.at(entry_key);
    if (!entry.weights || entry.protected_entry || protected_budget_bytes_ == 0) {
        return;
    }
    while (protected_bytes_ + entry.bytes > protected_budget_bytes_) {
        const std::size_t before = protected_bytes_;
        demote_coldest_protected(entry_key);
        if (protected_bytes_ == before) return;
    }
    entry.protected_entry = true;
    protected_bytes_ += entry.bytes;
}

bool CpuExpertCache::make_room(std::size_t incoming_bytes,
                               std::uint64_t except_key) {
    if (incoming_bytes > budget_bytes_) return false;
    while (resident_bytes_ + incoming_bytes > budget_bytes_) {
        std::uint64_t victim = choose_victim(false, except_key);
        if (victim == kNoKey) victim = choose_victim(true, except_key);
        if (victim == kNoKey) return false;

        auto it = entries_.find(victim);
        Entry& entry = it->second;
        resident_bytes_ -= entry.bytes;
        if (entry.protected_entry) protected_bytes_ -= entry.bytes;
        entries_.erase(it);
        ++evictions_;
    }
    return true;
}

std::shared_ptr<const CpuExpertWeights> CpuExpertCache::acquire(
    int layer, int expert, const Loader& loader) {
    const std::uint64_t entry_key = key(layer, expert);
    std::shared_future<std::shared_ptr<const CpuExpertWeights>> pending;
    std::shared_ptr<std::promise<std::shared_ptr<const CpuExpertWeights>>> promise;

    {
        std::unique_lock lock(mutex_);
        ++tick_;
        decay_if_needed();
        Entry& entry = entries_[entry_key];
        entry.heat += 1.0;
        entry.last_used = tick_;

        if (entry.weights) {
            ++hits_;
            if (entry.observed_accesses <
                std::numeric_limits<std::uint32_t>::max()) {
                ++entry.observed_accesses;
            }
            if (entry.observed_accesses >= 2) {
                promote_to_protected(entry_key);
            }
            return entry.weights;
        }
        if (entry.loading.valid()) {
            ++coalesced_waits_;
            pending = entry.loading;
            lock.unlock();
            return pending.get();
        }

        ++misses_;
        promise = std::make_shared<
            std::promise<std::shared_ptr<const CpuExpertWeights>>>();
        entry.loading = promise->get_future().share();
    }

    std::shared_ptr<const CpuExpertWeights> loaded;
    try {
        loaded = std::make_shared<const CpuExpertWeights>(loader());
        if (loaded->w13.rows == 0 || loaded->w2.rows == 0) {
            throw std::runtime_error("CPU expert loader returned empty weights");
        }
    } catch (...) {
        {
            std::lock_guard lock(mutex_);
            entries_.erase(entry_key);
        }
        promise->set_exception(std::current_exception());
        throw;
    }

    {
        std::lock_guard lock(mutex_);
        auto it = entries_.find(entry_key);
        if (it == entries_.end()) {
            promise->set_value(loaded);
            return loaded;
        }
        Entry& entry = it->second;
        entry.loading = {};
        entry.bytes = loaded->memory_bytes();
        entry.observed_accesses = 1;
        if (budget_bytes_ > 0 && make_room(entry.bytes, entry_key)) {
            entry.weights = loaded;
            resident_bytes_ += entry.bytes;
        } else {
            entries_.erase(it);
        }
    }

    promise->set_value(loaded);
    return loaded;
}

std::size_t CpuExpertCache::resident_bytes() const {
    std::lock_guard lock(mutex_);
    return resident_bytes_;
}

std::size_t CpuExpertCache::hits() const {
    std::lock_guard lock(mutex_);
    return hits_;
}

std::size_t CpuExpertCache::misses() const {
    std::lock_guard lock(mutex_);
    return misses_;
}

std::size_t CpuExpertCache::coalesced_waits() const {
    std::lock_guard lock(mutex_);
    return coalesced_waits_;
}

std::size_t CpuExpertCache::evictions() const {
    std::lock_guard lock(mutex_);
    return evictions_;
}

bool CpuExpertCache::resident(int layer, int expert) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(key(layer, expert));
    return it != entries_.end() && static_cast<bool>(it->second.weights);
}

bool CpuExpertCache::protected_resident(int layer, int expert) const {
    std::lock_guard lock(mutex_);
    const auto it = entries_.find(key(layer, expert));
    return it != entries_.end() && it->second.weights &&
           it->second.protected_entry;
}

} // namespace celeg
