#include "lfm/runtime/cache/prefix_radix.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    lfm::PrefixRadixIndex index;
    LFM_TEST_CHECK(index.size() == 0);
    LFM_TEST_CHECK(index.node_count() == 1);
    LFM_TEST_CHECK(!index.longest_prefix({1, 2, 3}));

    index.insert({1, 2}, 10);
    index.insert({1, 2, 3, 4}, 20);
    index.insert({1, 5}, 30);
    LFM_TEST_CHECK(index.size() == 3);

    auto match = index.longest_prefix({1, 2, 3, 4, 9});
    LFM_TEST_CHECK(match && match->id == 20 && match->matched_tokens == 4);
    match = index.longest_prefix({1, 2, 8});
    LFM_TEST_CHECK(match && match->id == 10 && match->matched_tokens == 2);
    match = index.longest_prefix({1, 5, 7});
    LFM_TEST_CHECK(match && match->id == 30 && match->matched_tokens == 2);
    LFM_TEST_CHECK(!index.longest_prefix({9, 9}));

    LFM_TEST_CHECK(index.erase({1, 2, 3, 4}, 20));
    match = index.longest_prefix({1, 2, 3, 4, 9});
    LFM_TEST_CHECK(match && match->id == 10 && match->matched_tokens == 2);
    LFM_TEST_CHECK(!index.erase({1, 2, 3, 4}, 20));
    LFM_TEST_CHECK(index.erase({1, 2}, 10));
    LFM_TEST_CHECK(!index.longest_prefix({1, 2, 8}));
    LFM_TEST_CHECK(index.size() == 1);

    bool caught = false;
    try { index.insert({}, 99); } catch (const std::invalid_argument&) { caught = true; }
    LFM_TEST_CHECK(caught);
    caught = false;
    try { index.insert({1}, 0); } catch (const std::invalid_argument&) { caught = true; }
    LFM_TEST_CHECK(caught);

    index.clear();
    LFM_TEST_CHECK(index.size() == 0 && index.node_count() == 1);
    std::cout << "prefix_radix_test: ok\n";
}
