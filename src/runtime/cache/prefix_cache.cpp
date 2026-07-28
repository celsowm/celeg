#include "lfm/runtime/cache/prefix_cache.hpp"
#include "lfm/runtime/cache/page_lease.hpp"

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
    const auto match = index_.longest(prompt);
    if (!match) return nullptr;
    const auto found = entries_.find(match->id);
    if (found == entries_.end()) throw std::runtime_error("prefix index references a missing cache entry");
    if (matched_tokens) *matched_tokens = match->matched_tokens;
    return found->second.get();
}

PrefixCacheManager::Entry* PrefixCacheManager::exact_entry(
    const std::vector<int32_t>& prompt) {
    size_t matched = 0;
    Entry* entry = longest_entry(prompt, &matched);
    return entry && matched == prompt.size() ? entry : nullptr;
}

bool PrefixCacheManager::evict_one(PrefixCacheIndex::EntryId protected_id) {
    // O(log n) victim selection via the LRU index.
    const auto id = index_.oldest(protected_id);
    if (!id) return false;
    auto entry_it = entries_.find(*id);
    if (entry_it == entries_.end()) throw std::runtime_error("prefix index entry payload missing");
    PageLease<uint32_t> lease(
        entry_it->second->pages,
        [this](const std::vector<uint32_t>& pages) { pages_.release(pages); });
    index_.erase(*id);
    entries_.erase(*id);
    ++metrics_.evictions;
    return true;
}

bool PrefixCacheManager::evict_one() {
    return evict_one(0);
}

std::optional<std::vector<uint32_t>>
PrefixCacheManager::allocate_with_eviction(
    size_t token_count, PrefixCacheIndex::EntryId protected_id) {
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
    const auto indexed = index_.longest(prompt);
    if (!indexed) throw std::runtime_error("prefix index entry disappeared during acquire");
    if (!pages_.retain(entry->pages)) {
        throw std::runtime_error("failed to retain pages owned by prefix cache");
    }

    PageLease<uint32_t> page_lease(
        entry->pages,
        [this](const std::vector<uint32_t>& pages) { pages_.release(pages); });
    auto& pages = page_lease.pages();
    try {
        const size_t partial = matched_tokens %
            static_cast<size_t>(pages_.page_tokens());
        if (partial != 0 && !pages.empty()) {
            std::optional<uint32_t> cloned;
            for (;;) {
                cloned = pages_.clone_page_prefix(
                    pages.back(), static_cast<int>(partial));
                if (cloned || !evict_one(indexed->id)) break;
            }
            if (!cloned) {
                result.status = PrefixAcquireStatus::OutOfMemory;
                return result;
            }
            const uint32_t shared_last = pages.back();
            pages.back() = *cloned;
            pages_.release(std::vector<uint32_t>{shared_last});
            record_cow(partial);
        }

        const size_t total_pages_needed = reserved_tokens == 0 ? 0 :
            (reserved_tokens + static_cast<size_t>(pages_.page_tokens()) - 1) /
            static_cast<size_t>(pages_.page_tokens());
        const size_t additional_pages = total_pages_needed > pages.size()
            ? total_pages_needed - pages.size() : 0;
        auto extra = allocate_with_eviction(
            additional_pages * static_cast<size_t>(pages_.page_tokens()),
            indexed->id);
        if (!extra) {
            result.status = PrefixAcquireStatus::OutOfMemory;
            return result;
        }
        pages.insert(pages.end(), extra->begin(), extra->end());
    } catch (...) {
        throw;
    }

    // Touch: update LRU index.
    index_.touch(indexed->id);

    result.status = PrefixAcquireStatus::Hit;
    result.pages = page_lease.release_ownership();
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
        const auto indexed = index_.longest(prompt);
        if (!indexed) throw std::runtime_error("prefix index entry disappeared during update");
        index_.touch(indexed->id);
        return true;
    }

    std::vector<uint32_t> cache_pages(
        request_pages.begin(),
        request_pages.begin() + static_cast<ptrdiff_t>(prefix_page_count));
    if (!pages_.retain(cache_pages)) {
        throw std::runtime_error("failed retaining request pages for prefix cache");
    }

    PageLease<uint32_t> cache_lease(
        std::move(cache_pages),
        [this](const std::vector<uint32_t>& pages) { pages_.release(pages); });
    auto& owned_pages = cache_lease.pages();

    try {
        const size_t partial = prompt.size() %
            static_cast<size_t>(pages_.page_tokens());
        if (partial != 0 && !owned_pages.empty()) {
            std::optional<uint32_t> cloned;
            for (;;) {
                cloned = pages_.clone_page_prefix(
                    owned_pages.back(), static_cast<int>(partial));
                if (cloned || !evict_one()) break;
            }
            if (!cloned) {
                return false;
            }
            const uint32_t request_page = owned_pages.back();
            owned_pages.back() = *cloned;
            pages_.release(std::vector<uint32_t>{request_page});
            record_cow(partial);
        }

        auto entry = std::make_unique<Entry>();
        entry->pages = owned_pages;
        entry->state = std::make_shared<PrefixState>(std::move(state));
        const auto id = index_.insert(prompt);
        try {
            const auto [_, inserted] = entries_.emplace(id, std::move(entry));
            if (!inserted) throw std::runtime_error("duplicate prefix cache ID");
        } catch (...) {
            index_.erase(id);
            throw;
        }
        cache_lease.release_ownership();
        ++metrics_.inserts;
        while (entries_.size() > maximum_entries_) {
            if (!evict_one(id)) break;
        }
        return true;
    } catch (...) { throw; }
}

void PrefixCacheManager::clear() {
    for (auto& [_, entry] : entries_) {
        PageLease<uint32_t> lease(
            entry->pages,
            [this](const std::vector<uint32_t>& pages) { pages_.release(pages); });
    }
    entries_.clear();
    index_.clear();
}

} // namespace lfm
