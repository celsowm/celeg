#include "celeg/runtime/cache/prefix_cache_index.hpp"

#include <limits>
#include <stdexcept>

namespace celeg {

PrefixCacheIndex::EntryId PrefixCacheIndex::insert(
    const std::vector<int32_t>& prompt) {
    if (prompt.empty()) throw std::invalid_argument("cannot index an empty prefix");
    if (next_id_ == std::numeric_limits<EntryId>::max()) {
        throw std::overflow_error("prefix cache entry ID space exhausted");
    }
    Entry entry;
    entry.id = next_id_++;
    entry.prompt = prompt;
    entry.last_used = ++clock_;
    radix_.insert(entry.prompt, entry.id);
    try {
        entries_.emplace(entry.id, entry);
        lru_.emplace(entry.last_used, entry.id);
    } catch (...) {
        radix_.erase(entry.prompt, entry.id);
        entries_.erase(entry.id);
        lru_.erase({entry.last_used, entry.id});
        throw;
    }
    return entry.id;
}

std::optional<PrefixCacheIndex::Match> PrefixCacheIndex::longest(
    const std::vector<int32_t>& prompt) const {
    const auto match = radix_.longest_prefix(prompt);
    if (!match) return std::nullopt;
    const auto found = entries_.find(match->id);
    if (found == entries_.end() ||
        found->second.prompt.size() != match->matched_tokens) {
        throw std::runtime_error("prefix radix references a missing cache entry");
    }
    return Match{match->id, match->matched_tokens};
}

bool PrefixCacheIndex::erase(EntryId id) {
    const auto found = entries_.find(id);
    if (found == entries_.end()) return false;
    if (!radix_.erase(found->second.prompt, id)) {
        throw std::runtime_error("prefix radix lost an indexed entry");
    }
    lru_.erase({found->second.last_used, id});
    entries_.erase(found);
    return true;
}

void PrefixCacheIndex::touch(EntryId id) {
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::out_of_range("unknown prefix entry");
    lru_.erase({found->second.last_used, id});
    found->second.last_used = ++clock_;
    lru_.emplace(found->second.last_used, id);
}

std::optional<PrefixCacheIndex::EntryId> PrefixCacheIndex::oldest(
    EntryId protected_id) const {
    for (const auto& [_, id] : lru_) {
        if (id != protected_id && entries_.contains(id)) return id;
    }
    return std::nullopt;
}

void PrefixCacheIndex::clear() {
    entries_.clear();
    lru_.clear();
    radix_.clear();
}

const std::vector<int32_t>& PrefixCacheIndex::prompt(EntryId id) const {
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::out_of_range("unknown prefix entry");
    return found->second.prompt;
}

}
