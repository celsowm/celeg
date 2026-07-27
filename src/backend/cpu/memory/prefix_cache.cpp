#include "lfm/backend/cpu/prefix_cache.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace lfm {

size_t CpuPrefixSnapshot::host_bytes() const {
    size_t bytes = logits.size() * sizeof(float) + seen_tokens.size();
    for (const auto& state : convolution_states) bytes += state.size() * sizeof(float);
    for (const auto& pages : attention_pages) bytes += pages.size() * sizeof(CpuKvPageId);
    bytes += attention_token_counts.size() * sizeof(size_t);
    return bytes;
}

CpuPrefixCacheManager::CpuPrefixCacheManager(
    std::vector<std::shared_ptr<CpuKvPagePool>> pools,
    size_t maximum_entries,
    size_t maximum_bytes)
    : pools_(std::move(pools)), maximum_entries_(maximum_entries),
      maximum_bytes_(maximum_bytes) {
    if (pools_.empty()) throw std::invalid_argument("CPU prefix cache requires KV pools");
    if (maximum_entries_ == 0 || maximum_bytes_ == 0) {
        throw std::invalid_argument("CPU prefix cache limits must be positive");
    }
}

CpuPrefixCacheManager::~CpuPrefixCacheManager() { clear(); }

void CpuPrefixCacheManager::retain_snapshot(const CpuPrefixSnapshot& snapshot) {
    if (snapshot.attention_pages.size() != pools_.size()) {
        throw std::invalid_argument("CPU prefix snapshot layer count mismatch");
    }
    size_t retained_layers = 0;
    try {
        for (; retained_layers < pools_.size(); ++retained_layers) {
            for (CpuKvPageId page : snapshot.attention_pages[retained_layers]) {
                pools_[retained_layers]->retain(page);
            }
        }
    } catch (...) {
        for (size_t layer = 0; layer <= retained_layers && layer < pools_.size(); ++layer) {
            for (CpuKvPageId page : snapshot.attention_pages[layer]) {
                try {
                    pools_[layer]->release(page);
                } catch (const std::exception& error) {
                    std::clog << "CPU prefix cache rollback failed: "
                              << error.what() << '\n';
                } catch (...) {
                    std::clog << "CPU prefix cache rollback failed: unknown exception\n";
                }
            }
        }
        throw;
    }
}

void CpuPrefixCacheManager::release_snapshot(const CpuPrefixSnapshot& snapshot) noexcept {
    const size_t layers = std::min(snapshot.attention_pages.size(), pools_.size());
    for (size_t layer = 0; layer < layers; ++layer) {
        for (CpuKvPageId page : snapshot.attention_pages[layer]) {
            try {
                pools_[layer]->release(page);
            } catch (const std::exception& error) {
                std::clog << "CPU prefix cache release failed: " << error.what() << '\n';
            } catch (...) {
                std::clog << "CPU prefix cache release failed: unknown exception\n";
            }
        }
    }
}

size_t CpuPrefixCacheManager::snapshot_resident_bytes(
    const CpuPrefixSnapshot& snapshot) const {
    size_t bytes = snapshot.host_bytes();
    const size_t layers = std::min(snapshot.attention_pages.size(), pools_.size());
    for (size_t layer = 0; layer < layers; ++layer) {
        bytes += snapshot.attention_pages[layer].size() * pools_[layer]->page_bytes();
    }
    return bytes;
}

CpuPrefixCacheManager::Entry* CpuPrefixCacheManager::longest_entry(
    const std::vector<int32_t>& prompt, size_t* matched_tokens) {
    ++metrics_.radix_lookups;
    const auto match = radix_.longest_prefix(prompt);
    if (!match) return nullptr;
    auto found = entries_.find(match->id);
    if (found == entries_.end()) {
        throw std::runtime_error("CPU prefix radix references a missing entry");
    }
    if (matched_tokens) *matched_tokens = match->matched_tokens;
    return found->second.get();
}

CpuPrefixCacheManager::Entry* CpuPrefixCacheManager::exact_entry(
    const std::vector<int32_t>& prompt) {
    size_t matched = 0;
    Entry* entry = longest_entry(prompt, &matched);
    return entry && matched == prompt.size() ? entry : nullptr;
}

CpuPrefixSnapshot CpuPrefixCacheManager::clone_snapshot(
    const CpuPrefixSnapshot& source, size_t prefix_tokens,
    int destination_numa_node) {
    CpuPrefixSnapshot result = source;
    result.numa_node = destination_numa_node;
    result.attention_pages.assign(source.attention_pages.size(), {});
    const size_t partial = pools_.front()->page_tokens() == 0 ? 0 :
        prefix_tokens % pools_.front()->page_tokens();
    try {
        for (size_t layer = 0; layer < pools_.size(); ++layer) {
            const auto& source_pages = source.attention_pages.at(layer);
            auto& destination_pages = result.attention_pages[layer];
            destination_pages = source_pages;
            for (CpuKvPageId page : destination_pages) pools_[layer]->retain(page);
            if (partial != 0 && !destination_pages.empty()) {
                const CpuKvPageId shared_tail = destination_pages.back();
                const CpuKvPageId private_tail = pools_[layer]->clone_prefix(
                    shared_tail, partial, destination_numa_node);
                destination_pages.back() = private_tail;
                pools_[layer]->release(shared_tail);
                ++metrics_.cow_pages;
                metrics_.cow_bytes += partial * pools_[layer]->kv_width() * 2 *
                    (pools_[layer]->mode() == CpuKvCacheMode::Fp32
                        ? sizeof(float) : sizeof(uint16_t));
            }
        }
        return result;
    } catch (...) {
        release_snapshot(result);
        throw;
    }
}

std::optional<CpuPrefixMatch> CpuPrefixCacheManager::acquire(
    const std::vector<int32_t>& prompt, int request_numa_node) {
    size_t matched = 0;
    Entry* entry = longest_entry(prompt, &matched);
    if (!entry) {
        ++metrics_.misses;
        return std::nullopt;
    }
    CpuPrefixMatch result;
    result.matched_tokens = matched;
    result.snapshot = clone_snapshot(entry->snapshot, matched, request_numa_node);
    lru_index_.erase({entry->last_used, entry->id});
    entry->last_used = ++clock_;
    lru_index_.insert({entry->last_used, entry->id});
    ++metrics_.hits;
    metrics_.reused_tokens += matched;
    if (matched != prompt.size()) ++metrics_.partial_hits;
    return result;
}

bool CpuPrefixCacheManager::evict_one(PrefixRadixIndex::EntryId protected_id) {
    // O(log n) victim selection via the LRU index.
    for (auto it = lru_index_.begin(); it != lru_index_.end(); ++it) {
        if (it->second == protected_id) continue;
        const auto id = it->second;
        auto entry_it = entries_.find(id);
        if (entry_it == entries_.end()) continue;
        Entry* victim = entry_it->second.get();
        lru_index_.erase(it);
        const size_t bytes = victim->resident_bytes;
        release_snapshot(victim->snapshot);
        if (!radix_.erase(victim->prompt, id)) {
            throw std::runtime_error("CPU prefix radix eviction lost its entry");
        }
        entries_.erase(id);
        metrics_.resident_bytes = metrics_.resident_bytes > bytes
            ? metrics_.resident_bytes - bytes : 0;
        ++metrics_.evictions;
        return true;
    }
    return false;
}

bool CpuPrefixCacheManager::evict_one() { return evict_one(0); }

bool CpuPrefixCacheManager::insert(
    const std::vector<int32_t>& prompt,
    const CpuPrefixSnapshot& request_snapshot) {
    if (prompt.empty()) return false;
    if (request_snapshot.position != prompt.size()) {
        throw std::invalid_argument("CPU prefix snapshot position does not match prompt");
    }
    if (Entry* existing = exact_entry(prompt)) {
        CpuPrefixSnapshot replacement = clone_snapshot(
            request_snapshot, prompt.size(), request_snapshot.numa_node);
        const size_t replacement_bytes = snapshot_resident_bytes(replacement);
        const size_t old_bytes = existing->resident_bytes;
        release_snapshot(existing->snapshot);
        existing->snapshot = std::move(replacement);
        existing->resident_bytes = replacement_bytes;
        lru_index_.erase({existing->last_used, existing->id});
        existing->last_used = ++clock_;
        lru_index_.insert({existing->last_used, existing->id});
        metrics_.resident_bytes = metrics_.resident_bytes - old_bytes + replacement_bytes;
        while (metrics_.resident_bytes > maximum_bytes_) {
            if (!evict_one(existing->id)) break;
        }
        return true;
    }
    if (next_id_ == std::numeric_limits<PrefixRadixIndex::EntryId>::max()) {
        throw std::overflow_error("CPU prefix cache entry ID space exhausted");
    }
    auto entry = std::make_unique<Entry>();
    entry->id = next_id_++;
    entry->prompt = prompt;
    entry->snapshot = clone_snapshot(
        request_snapshot, prompt.size(), request_snapshot.numa_node);
    entry->resident_bytes = snapshot_resident_bytes(entry->snapshot);
    entry->last_used = ++clock_;
    const auto id = entry->id;
    const auto lru_key = std::pair{entry->last_used, id};
    lru_index_.insert(lru_key);
    radix_.insert(entry->prompt, id);
    try {
        entries_.emplace(id, std::move(entry));
    } catch (...) {
        lru_index_.erase(lru_key);
        radix_.erase(prompt, id);
        throw;
    }
    metrics_.resident_bytes += entries_.at(id)->resident_bytes;
    ++metrics_.inserts;
    while (entries_.size() > maximum_entries_ || metrics_.resident_bytes > maximum_bytes_) {
        if (!evict_one(id)) {
            auto inserted = entries_.find(id);
            if (inserted != entries_.end()) {
                const size_t bytes = inserted->second->resident_bytes;
                lru_index_.erase({inserted->second->last_used, id});
                release_snapshot(inserted->second->snapshot);
                radix_.erase(inserted->second->prompt, id);
                entries_.erase(inserted);
                metrics_.resident_bytes = metrics_.resident_bytes > bytes
                    ? metrics_.resident_bytes - bytes : 0;
            }
            return false;
        }
    }
    return entries_.find(id) != entries_.end();
}

void CpuPrefixCacheManager::clear() {
    for (auto& [_, entry] : entries_) release_snapshot(entry->snapshot);
    entries_.clear();
    lru_index_.clear();
    radix_.clear();
    metrics_.resident_bytes = 0;
}

} // namespace lfm
