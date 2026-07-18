#include "lfm/prefix_cache.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lfm {

PrefixCacheManager::PrefixCacheManager(IKvPageAllocator& pages,
                                       bool enabled,
                                       size_t maximum_entries)
    : pages_(pages), enabled_(enabled), maximum_entries_(maximum_entries) {
    if (enabled_ && maximum_entries_ == 0) {
        throw std::invalid_argument(
            "prefix cache maximum_entries must be positive when enabled");
    }
}

PrefixCacheManager::~PrefixCacheManager() {
    clear();
}

PrefixCacheManager::Entry* PrefixCacheManager::longest_entry(
    const std::vector<int32_t>& prompt, size_t* matched_tokens) {
    if (!enabled_) return nullptr;
    ++metrics_.radix_lookups;
    const auto match = radix_.longest_prefix(prompt);
    if (!match) return nullptr;
    const auto found = entries_.find(match->id);
    if (found == entries_.end() ||
        found->second->prompt.size() != match->matched_tokens) {
        throw std::runtime_error("prefix radix references a missing cache entry");
    }
    if (matched_tokens) *matched_tokens = match->matched_tokens;
    return found->second.get();
}

PrefixCacheManager::Entry* PrefixCacheManager::exact_entry(
    const std::vector<int32_t>& prompt) {
    size_t matched = 0;
    Entry* entry = longest_entry(prompt, &matched);
    return entry && matched == prompt.size() ? entry : nullptr;
}

bool PrefixCacheManager::evict_one(PrefixRadixIndex::EntryId protected_id) {
    Entry* victim = nullptr;
    for (auto& [id, entry] : entries_) {
        if (id == protected_id) continue;
        if (!victim || entry->last_used < victim->last_used) {
            victim = entry.get();
        }
    }
    if (!victim) return false;
    const auto id = victim->id;
    pages_.release(victim->pages);
    if (!radix_.erase(victim->prompt, id)) {
        throw std::runtime_error("prefix radix eviction lost its entry");
    }
    entries_.erase(id);
    ++metrics_.evictions;
    return true;
}

bool PrefixCacheManager::evict_one() {
    return evict_one(0);
}

std::optional<std::vector<uint32_t>>
PrefixCacheManager::allocate_with_eviction(
    size_t token_count, PrefixRadixIndex::EntryId protected_id) {
    for (;;) {
        auto allocated = pages_.allocate_tokens(token_count);
        if (allocated) return allocated;
        if (!enabled_ || !evict_one(protected_id)) return std::nullopt;
    }
}

std::optional<std::vector<uint32_t>>
PrefixCacheManager::allocate_request_pages(size_t token_count) {
    return allocate_with_eviction(token_count);
}

void PrefixCacheManager::record_cow(size_t used_tokens) {
    ++metrics_.cow_pages;
    const size_t full = pages_.page_bytes();
    const size_t copied = full * used_tokens /
        static_cast<size_t>(pages_.page_tokens());
    metrics_.cow_bytes_copied += copied;
    metrics_.cow_bytes_saved += full - copied;
}

PrefixAcquireResult PrefixCacheManager::acquire(
    const std::vector<int32_t>& prompt, size_t reserved_tokens) {
    PrefixAcquireResult result;
    if (!enabled_) {
        result.status = PrefixAcquireStatus::Disabled;
        return result;
    }

    size_t matched_tokens = 0;
    Entry* entry = longest_entry(prompt, &matched_tokens);
    if (!entry) {
        ++metrics_.misses;
        result.status = PrefixAcquireStatus::Miss;
        return result;
    }
    if (!pages_.retain(entry->pages)) {
        throw std::runtime_error("failed to retain pages owned by prefix cache");
    }

    result.pages = entry->pages;
    try {
        const size_t partial = matched_tokens %
            static_cast<size_t>(pages_.page_tokens());
        if (partial != 0 && !result.pages.empty()) {
            std::optional<uint32_t> cloned;
            for (;;) {
                cloned = pages_.clone_page_prefix(
                    result.pages.back(), static_cast<int>(partial));
                if (cloned || !evict_one(entry->id)) break;
            }
            if (!cloned) {
                pages_.release(result.pages);
                result.pages.clear();
                result.status = PrefixAcquireStatus::OutOfMemory;
                return result;
            }
            const uint32_t shared_last = result.pages.back();
            result.pages.back() = *cloned;
            pages_.release(std::vector<uint32_t>{shared_last});
            record_cow(partial);
        }

        const size_t total_pages_needed = reserved_tokens == 0 ? 0 :
            (reserved_tokens + static_cast<size_t>(pages_.page_tokens()) - 1) /
            static_cast<size_t>(pages_.page_tokens());
        const size_t additional_pages = total_pages_needed > result.pages.size()
            ? total_pages_needed - result.pages.size() : 0;
        auto extra = allocate_with_eviction(
            additional_pages * static_cast<size_t>(pages_.page_tokens()),
            entry->id);
        if (!extra) {
            pages_.release(result.pages);
            result.pages.clear();
            result.status = PrefixAcquireStatus::OutOfMemory;
            return result;
        }
        result.pages.insert(result.pages.end(), extra->begin(), extra->end());
    } catch (...) {
        if (!result.pages.empty()) pages_.release(result.pages);
        throw;
    }

    entry->last_used = ++clock_;
    result.status = PrefixAcquireStatus::Hit;
    result.matched_tokens = matched_tokens;
    result.state = entry->state;
    ++metrics_.hits;
    metrics_.reused_tokens += matched_tokens;
    if (matched_tokens != prompt.size()) ++metrics_.partial_hits;
    return result;
}

bool PrefixCacheManager::insert_or_update(
    const std::vector<int32_t>& prompt,
    const std::vector<uint32_t>& request_pages,
    PrefixState state) {
    if (!enabled_) return false;
    if (prompt.empty()) {
        throw std::invalid_argument("cannot cache an empty prefix");
    }
    const size_t prefix_page_count =
        (prompt.size() + static_cast<size_t>(pages_.page_tokens()) - 1) /
        static_cast<size_t>(pages_.page_tokens());
    if (request_pages.size() < prefix_page_count) {
        throw std::invalid_argument("request page table is shorter than prefix");
    }

    if (Entry* existing = exact_entry(prompt)) {
        existing->state = std::make_shared<PrefixState>(std::move(state));
        existing->last_used = ++clock_;
        return true;
    }

    std::vector<uint32_t> cache_pages(
        request_pages.begin(),
        request_pages.begin() + static_cast<ptrdiff_t>(prefix_page_count));
    if (!pages_.retain(cache_pages)) {
        throw std::runtime_error("failed retaining request pages for prefix cache");
    }

    try {
        const size_t partial = prompt.size() %
            static_cast<size_t>(pages_.page_tokens());
        if (partial != 0 && !cache_pages.empty()) {
            std::optional<uint32_t> cloned;
            for (;;) {
                cloned = pages_.clone_page_prefix(
                    cache_pages.back(), static_cast<int>(partial));
                if (cloned || !evict_one()) break;
            }
            if (!cloned) {
                pages_.release(cache_pages);
                return false;
            }
            const uint32_t request_page = cache_pages.back();
            cache_pages.back() = *cloned;
            pages_.release(std::vector<uint32_t>{request_page});
            record_cow(partial);
        }

        if (next_id_ == std::numeric_limits<PrefixRadixIndex::EntryId>::max()) {
            throw std::overflow_error("prefix cache entry ID space exhausted");
        }
        auto entry = std::make_unique<Entry>();
        entry->id = next_id_++;
        entry->prompt = prompt;
        entry->pages = cache_pages;
        entry->state = std::make_shared<PrefixState>(std::move(state));
        entry->last_used = ++clock_;
        const auto id = entry->id;
        radix_.insert(entry->prompt, id);
        try {
            const auto [_, inserted] = entries_.emplace(id, std::move(entry));
            if (!inserted) throw std::runtime_error("duplicate prefix cache ID");
        } catch (...) {
            radix_.erase(prompt, id);
            throw;
        }
        cache_pages.clear();
        ++metrics_.inserts;
        while (entries_.size() > maximum_entries_) {
            if (!evict_one(id)) break;
        }
        return true;
    } catch (...) {
        if (!cache_pages.empty()) pages_.release(cache_pages);
        throw;
    }
}

void PrefixCacheManager::clear() {
    for (auto& [_, entry] : entries_) pages_.release(entry->pages);
    entries_.clear();
    radix_.clear();
}

} // namespace lfm
