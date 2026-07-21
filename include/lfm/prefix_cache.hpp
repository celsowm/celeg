#pragma once

#include "lfm/kv_page_allocator.hpp"
#include "lfm/prefix_radix.hpp"
#include "lfm/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace lfm {

struct PrefixCacheMetrics {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t inserts = 0;
    uint64_t evictions = 0;
    uint64_t partial_hits = 0;
    uint64_t reused_tokens = 0;
    uint64_t cow_pages = 0;
    uint64_t cow_bytes_copied = 0;
    uint64_t cow_bytes_saved = 0;
    uint64_t radix_lookups = 0;
};

enum class PrefixAcquireStatus : uint8_t {
    Disabled,
    Miss,
    Hit,
    OutOfMemory,
};

struct PrefixAcquireResult {
    PrefixAcquireStatus status = PrefixAcquireStatus::Disabled;
    size_t matched_tokens = 0;
    std::vector<uint32_t> pages;
    std::shared_ptr<const PrefixState> state;
};

// Owns prefix-cache indexing, LRU, page references and COW. Calls are expected
// to be externally synchronized by ConcurrentEngine.
class PrefixCacheManager {
public:
    PrefixCacheManager(IKvPageAllocator& pages,
                       bool enabled,
                       size_t maximum_entries);
    ~PrefixCacheManager();

    PrefixCacheManager(const PrefixCacheManager&) = delete;
    PrefixCacheManager& operator=(const PrefixCacheManager&) = delete;

    PrefixAcquireResult acquire(const std::vector<int32_t>& prompt,
                                size_t reserved_tokens);
    std::optional<std::vector<uint32_t>> allocate_request_pages(
        size_t token_count);
    bool insert_or_update(const std::vector<int32_t>& prompt,
                          const std::vector<uint32_t>& request_pages,
                          PrefixState state);
    bool evict_one();
    void clear();

    bool enabled() const { return enabled_; }
    size_t size() const { return entries_.size(); }
    size_t radix_nodes() const { return radix_.node_count(); }
    const PrefixCacheMetrics& metrics() const { return metrics_; }

private:
    struct Entry {
        PrefixRadixIndex::EntryId id = 0;
        std::vector<int32_t> prompt;
        std::vector<uint32_t> pages;
        std::shared_ptr<const PrefixState> state;
        uint64_t last_used = 0;
    };

    Entry* exact_entry(const std::vector<int32_t>& prompt);
    Entry* longest_entry(const std::vector<int32_t>& prompt,
                         size_t* matched_tokens);
    std::optional<std::vector<uint32_t>> allocate_with_eviction(
        size_t token_count,
        PrefixRadixIndex::EntryId protected_id = 0);
    bool evict_one(PrefixRadixIndex::EntryId protected_id);
    void record_cow(size_t used_tokens);

    IKvPageAllocator& pages_;
    bool enabled_ = false;
    size_t maximum_entries_ = 0;
    PrefixRadixIndex radix_;
    std::unordered_map<PrefixRadixIndex::EntryId, std::unique_ptr<Entry>> entries_;
    // LRU index: (last_used, id) ascending — begin() is the oldest.
    std::set<std::pair<uint64_t, PrefixRadixIndex::EntryId>> lru_index_;
    PrefixRadixIndex::EntryId next_id_ = 1;
    uint64_t clock_ = 0;
    PrefixCacheMetrics metrics_;
};

} // namespace lfm
