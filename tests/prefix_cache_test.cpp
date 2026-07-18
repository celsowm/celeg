#include "lfm/prefix_cache.hpp"

#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

class FakePages final : public lfm::IKvPageAllocator {
public:
    FakePages(size_t count, int page_tokens)
        : page_tokens_(page_tokens), refs_(count, 0) {
        for (size_t i = count; i > 0; --i) free_.push_back(static_cast<uint32_t>(i - 1));
    }

    std::optional<std::vector<uint32_t>> allocate_tokens(size_t token_count) override {
        const size_t needed = token_count == 0 ? 0 :
            (token_count + static_cast<size_t>(page_tokens_) - 1) /
            static_cast<size_t>(page_tokens_);
        if (needed > free_.size()) return std::nullopt;
        std::vector<uint32_t> result;
        for (size_t i = 0; i < needed; ++i) {
            const uint32_t page = free_.back();
            free_.pop_back();
            refs_[page] = 1;
            result.push_back(page);
        }
        return result;
    }

    bool retain(const std::vector<uint32_t>& pages) override {
        for (uint32_t page : pages) {
            if (page >= refs_.size() || refs_[page] == 0 ||
                refs_[page] == std::numeric_limits<uint32_t>::max()) return false;
        }
        for (uint32_t page : pages) ++refs_[page];
        return true;
    }

    void release(const std::vector<uint32_t>& pages) override {
        for (uint32_t page : pages) {
            if (page >= refs_.size() || refs_[page] == 0) {
                throw std::runtime_error("double release");
            }
        }
        for (uint32_t page : pages) {
            if (--refs_[page] == 0) free_.push_back(page);
        }
    }

    std::optional<uint32_t> clone_page_prefix(uint32_t source,
                                               int used_tokens) override {
        if (source >= refs_.size() || refs_[source] == 0 ||
            used_tokens <= 0 || used_tokens > page_tokens_) {
            throw std::invalid_argument("bad clone");
        }
        auto page = allocate_tokens(static_cast<size_t>(page_tokens_));
        if (!page) return std::nullopt;
        ++clones;
        last_clone_tokens = used_tokens;
        return page->front();
    }

    size_t total_pages() const override { return refs_.size(); }
    size_t used_pages() const override { return refs_.size() - free_.size(); }
    int page_tokens() const override { return page_tokens_; }
    size_t page_bytes() const override { return 1024; }

    int clones = 0;
    int last_clone_tokens = 0;

private:
    int page_tokens_;
    std::vector<uint32_t> refs_;
    std::vector<uint32_t> free_;
};

} // namespace

int main() {
    using namespace lfm;

    FakePages pages(16, 4);
    PrefixCacheManager cache(pages, true, 2);

    auto request_pages = cache.allocate_request_pages(8);
    assert(request_pages && request_pages->size() == 2);

    PrefixState state;
    state.position = 3;
    state.seen_tokens = {1, 0, 1};
    state.logits_bf16 = {4, 5};
    state.conv_state_bf16 = {6};
    assert(cache.insert_or_update({10, 20, 30}, *request_pages, state));
    assert(cache.size() == 1);
    assert(pages.clones == 1);
    assert(pages.last_clone_tokens == 3);

    PrefixAcquireResult hit = cache.acquire({10, 20, 30, 40, 50}, 8);
    assert(hit.status == PrefixAcquireStatus::Hit);
    assert(hit.matched_tokens == 3);
    assert(hit.pages.size() == 2);
    assert(hit.state && hit.state->position == 3);
    assert(pages.clones == 2);

    PrefixAcquireResult miss = cache.acquire({99, 100}, 4);
    assert(miss.status == PrefixAcquireStatus::Miss);

    const PrefixCacheMetrics metrics = cache.metrics();
    assert(metrics.hits == 1);
    assert(metrics.partial_hits == 1);
    assert(metrics.misses == 1);
    assert(metrics.inserts == 1);
    assert(metrics.cow_pages == 2);
    assert(metrics.cow_bytes_saved > 0);

    pages.release(hit.pages);
    pages.release(*request_pages);
    cache.clear();
    assert(pages.used_pages() == 0);

    std::cout << "prefix_cache_test: ok\n";
    return 0;
}
