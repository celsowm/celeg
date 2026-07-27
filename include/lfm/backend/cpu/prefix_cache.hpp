#pragma once

#include "lfm/backend/cpu/paged_kv.hpp"
#include "lfm/runtime/cache/prefix_radix.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace lfm {

struct CpuPrefixSnapshot {
    size_t position = 0;
    int numa_node = -1;
    std::vector<std::vector<CpuKvPageId>> attention_pages;
    std::vector<size_t> attention_token_counts;
    std::vector<std::vector<float>> convolution_states;
    std::vector<float> logits;
    std::vector<uint8_t> seen_tokens;

    size_t host_bytes() const;
};

struct CpuPrefixCacheMetrics {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t partial_hits = 0;
    uint64_t inserts = 0;
    uint64_t evictions = 0;
    uint64_t reused_tokens = 0;
    uint64_t cow_pages = 0;
    uint64_t cow_bytes = 0;
    uint64_t radix_lookups = 0;
    size_t resident_bytes = 0;
};

struct CpuPrefixMatch {
    size_t matched_tokens = 0;
    CpuPrefixSnapshot snapshot;
};

// Externally synchronized prefix cache. It owns one retained reference for
// every cached page. acquire() returns a second, request-owned reference and
// clones the partial tail page before it can be modified.
class CpuPrefixCacheManager {
public:
    CpuPrefixCacheManager(
        std::vector<std::shared_ptr<CpuKvPagePool>> pools,
        size_t maximum_entries,
        size_t maximum_bytes);
    ~CpuPrefixCacheManager();

    CpuPrefixCacheManager(const CpuPrefixCacheManager&) = delete;
    CpuPrefixCacheManager& operator=(const CpuPrefixCacheManager&) = delete;

    std::optional<CpuPrefixMatch> acquire(const std::vector<int32_t>& prompt,
                                          int request_numa_node = -1);
    bool insert(const std::vector<int32_t>& prompt,
                const CpuPrefixSnapshot& request_snapshot);
    bool evict_one();
    void clear();

    size_t size() const { return entries_.size(); }
    size_t radix_nodes() const { return radix_.node_count(); }
    const CpuPrefixCacheMetrics& metrics() const { return metrics_; }

private:
    struct Entry {
        PrefixRadixIndex::EntryId id = 0;
        std::vector<int32_t> prompt;
        CpuPrefixSnapshot snapshot;
        uint64_t last_used = 0;
        size_t resident_bytes = 0;
    };

    void retain_snapshot(const CpuPrefixSnapshot& snapshot);
    void release_snapshot(const CpuPrefixSnapshot& snapshot) noexcept;
    CpuPrefixSnapshot clone_for_request(const CpuPrefixSnapshot& cached,
                                        size_t matched_tokens,
                                        int request_numa_node);
    CpuPrefixSnapshot clone_for_cache(const CpuPrefixSnapshot& request,
                                      size_t prefix_tokens);
    Entry* exact_entry(const std::vector<int32_t>& prompt);
    Entry* longest_entry(const std::vector<int32_t>& prompt,
                         size_t* matched_tokens);
    bool evict_one(PrefixRadixIndex::EntryId protected_id);
    size_t snapshot_resident_bytes(const CpuPrefixSnapshot& snapshot) const;

    std::vector<std::shared_ptr<CpuKvPagePool>> pools_;
    size_t maximum_entries_ = 0;
    size_t maximum_bytes_ = 0;
    PrefixRadixIndex radix_;
    std::unordered_map<PrefixRadixIndex::EntryId, std::unique_ptr<Entry>> entries_;
    // LRU index: (last_used, id) ascending — begin() is the oldest.
    std::set<std::pair<uint64_t, PrefixRadixIndex::EntryId>> lru_index_;
    PrefixRadixIndex::EntryId next_id_ = 1;
    uint64_t clock_ = 0;
    CpuPrefixCacheMetrics metrics_;
};

} // namespace lfm
