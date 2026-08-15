#pragma once

#include "celeg/backend/cpu/gguf.hpp"
#include "celeg/runtime/cache/expert_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace celeg {

struct CpuExpertWeights {
    CpuLinearWeight w13;
    CpuLinearWeight w2;

    std::size_t memory_bytes() const {
        return w13.memory_bytes() + w2.memory_bytes();
    }
};

class CpuExpertCache {
public:
    using Loader = std::function<CpuExpertWeights()>;

    explicit CpuExpertCache(std::size_t budget_bytes);

    std::shared_ptr<const CpuExpertWeights> acquire(
        int layer, int expert, const Loader& loader);

    std::size_t budget_bytes() const { return budget_bytes_; }
    std::size_t resident_bytes() const;
    std::size_t hits() const;
    std::size_t misses() const;
    std::size_t coalesced_waits() const;
    std::size_t evictions() const;
    bool resident(int layer, int expert) const;
    bool protected_resident(int layer, int expert) const;

private:
    struct Entry {
        std::shared_ptr<const CpuExpertWeights> weights;
        std::shared_future<std::shared_ptr<const CpuExpertWeights>> loading;
        std::size_t bytes = 0;
        double heat = 0.0;
        std::uint64_t last_used = 0;
        std::uint32_t observed_accesses = 0;
        bool protected_entry = false;
    };

    static constexpr std::uint64_t kDecayInterval = 256;
    static constexpr double kDecayFactor = 0.5;
    static constexpr std::uint64_t kNoKey =
        std::numeric_limits<std::uint64_t>::max();

    static std::uint64_t key(int layer, int expert);
    void decay_if_needed();
    void promote_to_protected(std::uint64_t entry_key);
    void demote_coldest_protected(std::uint64_t except_key);
    bool make_room(std::size_t incoming_bytes, std::uint64_t except_key);
    std::uint64_t choose_victim(bool protected_entries,
                                std::uint64_t except_key) const;
    double priority(const Entry& entry) const;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, Entry> entries_;
    std::size_t budget_bytes_ = 0;
    std::size_t protected_budget_bytes_ = 0;
    std::size_t resident_bytes_ = 0;
    std::size_t protected_bytes_ = 0;
    std::uint64_t tick_ = 0;
    std::uint64_t last_decay_tick_ = 0;
    std::size_t hits_ = 0;
    std::size_t misses_ = 0;
    std::size_t coalesced_waits_ = 0;
    std::size_t evictions_ = 0;
};

}
