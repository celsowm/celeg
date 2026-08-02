#include "celeg/runtime/concurrency/policy.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace celeg {

const char* request_status_name(RequestStatus status) {
    switch (status) {
        case RequestStatus::Queued: return "queued";
        case RequestStatus::Prefill: return "prefilling";
        case RequestStatus::Decoding: return "decoding";
        case RequestStatus::Finished: return "completed";
        case RequestStatus::Cancelled: return "cancelled";
        case RequestStatus::Failed: return "failed";
    }
    return "unknown";
}

PagedBlockPool::PagedBlockPool(size_t pages, int page_tokens)
    : page_tokens_(page_tokens), ref_counts_(pages, 0) {
    if (pages == 0) throw std::invalid_argument("paged pool needs pages");
    if (page_tokens <= 0) throw std::invalid_argument("page_tokens must be positive");
    if (pages > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("paged pool exceeds uint32 page IDs");
    }
    free_.reserve(pages);
    for (size_t i = pages; i > 0; --i) free_.push_back(static_cast<uint32_t>(i - 1));
}

std::optional<std::vector<uint32_t>> PagedBlockPool::allocate_tokens(size_t token_count) {
    const size_t width = static_cast<size_t>(page_tokens_);
    if (token_count > std::numeric_limits<size_t>::max() - (width - 1)) {
        throw std::overflow_error("paged pool token count overflow");
    }
    const size_t needed = token_count == 0 ? 0 :
        (token_count + width - 1) / width;
    if (needed > free_.size()) return std::nullopt;
    std::vector<uint32_t> result;
    result.reserve(needed);
    for (size_t i = 0; i < needed; ++i) {
        const uint32_t page = free_.back();
        free_.pop_back();
        ref_counts_[page] = 1;
        result.push_back(page);
    }
    return result;
}


bool PagedBlockPool::retain(const std::vector<uint32_t>& pages) {
    // Validate the whole transaction first, including duplicate page IDs.
    // O(k) sort + run-length encode instead of O(total_pages) histogram.
    if (pages.empty()) return true;
    std::vector<uint32_t> sorted = pages;
    std::sort(sorted.begin(), sorted.end());
    uint32_t prev = sorted[0];
    uint32_t count = 1;
    auto flush = [&](uint32_t page, uint32_t n) -> bool {
        if (page >= ref_counts_.size() || ref_counts_[page] == 0) return false;
        if (n > std::numeric_limits<uint32_t>::max() - ref_counts_[page]) return false;
        return true;
    };
    for (size_t i = 1; i <= sorted.size(); ++i) {
        if (i < sorted.size() && sorted[i] == prev) {
            ++count;
            continue;
        }
        if (!flush(prev, count)) return false;
        if (i < sorted.size()) { prev = sorted[i]; count = 1; }
    }
    // Apply: run-length decode and update ref counts.
    prev = sorted[0];
    count = 1;
    for (size_t i = 1; i <= sorted.size(); ++i) {
        if (i < sorted.size() && sorted[i] == prev) {
            ++count;
            continue;
        }
        ref_counts_[prev] += count;
        if (i < sorted.size()) { prev = sorted[i]; count = 1; }
    }
    return true;
}

void PagedBlockPool::release(const std::vector<uint32_t>& pages) {
    // Release is also transactional: an invalid page later in the vector must
    // not silently release earlier pages.  O(k) sort + run-length encode.
    if (pages.empty()) return;
    std::vector<uint32_t> sorted = pages;
    std::sort(sorted.begin(), sorted.end());
    uint32_t prev = sorted[0];
    uint32_t count = 1;
    // Validate first.
    for (size_t i = 1; i <= sorted.size(); ++i) {
        if (i < sorted.size() && sorted[i] == prev) {
            ++count;
            continue;
        }
        if (prev >= ref_counts_.size() || count > ref_counts_[prev]) {
            throw std::runtime_error("invalid or double page release");
        }
        if (i < sorted.size()) { prev = sorted[i]; count = 1; }
    }
    // Apply.
    prev = sorted[0];
    count = 1;
    for (size_t i = 1; i <= sorted.size(); ++i) {
        if (i < sorted.size() && sorted[i] == prev) {
            ++count;
            continue;
        }
        ref_counts_[prev] -= count;
        if (ref_counts_[prev] == 0) free_.push_back(prev);
        if (i < sorted.size()) { prev = sorted[i]; count = 1; }
    }
}

size_t PagedBlockPool::free_pages() const { return free_.size(); }

uint32_t PagedBlockPool::ref_count(uint32_t page) const {
    if (page >= ref_counts_.size()) {
        throw std::out_of_range("paged pool page is out of range");
    }
    return ref_counts_[page];
}

} // namespace celeg
