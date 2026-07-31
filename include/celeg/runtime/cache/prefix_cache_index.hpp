#pragma once

#include "celeg/runtime/cache/prefix_radix.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace celeg {

// Backend-neutral prefix identity and eviction policy. Payload ownership and
// page/snapshot lifetime remain in the CPU and CUDA cache managers.
class PrefixCacheIndex {
public:
    using EntryId = PrefixRadixIndex::EntryId;

    struct Match {
        EntryId id = 0;
        size_t matched_tokens = 0;
    };

    PrefixCacheIndex() = default;

    PrefixCacheIndex(const PrefixCacheIndex&) = delete;
    PrefixCacheIndex& operator=(const PrefixCacheIndex&) = delete;

    EntryId insert(const std::vector<int32_t>& prompt);
    std::optional<Match> longest(const std::vector<int32_t>& prompt) const;
    bool erase(EntryId id);
    void touch(EntryId id);
    std::optional<EntryId> oldest(EntryId protected_id = 0) const;
    void clear();

    size_t size() const { return entries_.size(); }
    size_t radix_nodes() const { return radix_.node_count(); }
    const std::vector<int32_t>& prompt(EntryId id) const;

private:
    struct Entry {
        EntryId id = 0;
        std::vector<int32_t> prompt;
        uint64_t last_used = 0;
    };

    PrefixRadixIndex radix_;
    std::unordered_map<EntryId, Entry> entries_;
    std::set<std::pair<uint64_t, EntryId>> lru_;
    EntryId next_id_ = 1;
    uint64_t clock_ = 0;
};

} // namespace celeg
