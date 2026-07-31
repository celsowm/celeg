#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace celeg {

// Move-only ownership guard for a retained page vector. The lease deliberately
// knows only how to release; callers perform backend-specific retain/clone
// operations before constructing it and can transfer ownership explicitly.
template <typename PageId>
class PageLease {
public:
    using Release = std::function<void(const std::vector<PageId>&)>;

    PageLease() = default;
    PageLease(std::vector<PageId> pages, Release release)
        : pages_(std::move(pages)), release_(std::move(release)) {}

    ~PageLease() { reset(); }

    PageLease(const PageLease&) = delete;
    PageLease& operator=(const PageLease&) = delete;
    PageLease(PageLease&& other) noexcept
        : pages_(std::move(other.pages_)), release_(std::move(other.release_)) {}
    PageLease& operator=(PageLease&& other) noexcept {
        if (this != &other) {
            reset();
            pages_ = std::move(other.pages_);
            release_ = std::move(other.release_);
        }
        return *this;
    }

    const std::vector<PageId>& pages() const { return pages_; }
    std::vector<PageId>& pages() { return pages_; }
    std::vector<PageId> release_ownership() noexcept {
        release_ = {};
        return std::move(pages_);
    }
    void release_now() {
        if (pages_.empty() || !release_) return;
        release_(pages_);
        pages_.clear();
        release_ = {};
    }
    void reset() noexcept {
        if (!pages_.empty() && release_) {
            try { release_(pages_); } catch (...) { /* no-throw destructor */ }
        }
        pages_.clear();
        release_ = {};
    }

private:
    std::vector<PageId> pages_;
    Release release_;
};

} // namespace celeg
