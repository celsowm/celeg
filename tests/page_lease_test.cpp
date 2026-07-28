#include "lfm/runtime/cache/page_lease.hpp"
#include "support/assertions.hpp"

#include <vector>

int main() {
    int releases = 0;
    std::vector<uint32_t> released;
    {
        lfm::PageLease<uint32_t> first(
            {1, 2}, [&](const std::vector<uint32_t>& pages) {
                ++releases;
                released = pages;
            });
        lfm::PageLease<uint32_t> second(std::move(first));
        LFM_TEST_CHECK(first.pages().empty());
        LFM_TEST_CHECK(second.pages().size() == 2);
    }
    LFM_TEST_CHECK(releases == 1);
    LFM_TEST_CHECK((released == std::vector<uint32_t>{1, 2}));

    releases = 0;
    {
        lfm::PageLease<uint32_t> lease(
            {3}, [&](const std::vector<uint32_t>&) { ++releases; });
        const auto owned = lease.release_ownership();
        LFM_TEST_CHECK(owned == std::vector<uint32_t>{3});
    }
    LFM_TEST_CHECK(releases == 0);
    return 0;
}
