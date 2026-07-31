#include "celeg/runtime/cache/prefix_radix.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    celeg::PrefixRadixIndex index;
    CELEG_TEST_CHECK(index.size() == 0);
    CELEG_TEST_CHECK(index.node_count() == 1);
    CELEG_TEST_CHECK(!index.longest_prefix({1, 2, 3}));

    index.insert({1, 2}, 10);
    index.insert({1, 2, 3, 4}, 20);
    index.insert({1, 5}, 30);
    CELEG_TEST_CHECK(index.size() == 3);

    auto match = index.longest_prefix({1, 2, 3, 4, 9});
    CELEG_TEST_CHECK(match && match->id == 20 && match->matched_tokens == 4);
    match = index.longest_prefix({1, 2, 8});
    CELEG_TEST_CHECK(match && match->id == 10 && match->matched_tokens == 2);
    match = index.longest_prefix({1, 5, 7});
    CELEG_TEST_CHECK(match && match->id == 30 && match->matched_tokens == 2);
    CELEG_TEST_CHECK(!index.longest_prefix({9, 9}));

    CELEG_TEST_CHECK(index.erase({1, 2, 3, 4}, 20));
    match = index.longest_prefix({1, 2, 3, 4, 9});
    CELEG_TEST_CHECK(match && match->id == 10 && match->matched_tokens == 2);
    CELEG_TEST_CHECK(!index.erase({1, 2, 3, 4}, 20));
    CELEG_TEST_CHECK(index.erase({1, 2}, 10));
    CELEG_TEST_CHECK(!index.longest_prefix({1, 2, 8}));
    CELEG_TEST_CHECK(index.size() == 1);

    bool caught = false;
    try { index.insert({}, 99); } catch (const std::invalid_argument&) { caught = true; }
    CELEG_TEST_CHECK(caught);
    caught = false;
    try { index.insert({1}, 0); } catch (const std::invalid_argument&) { caught = true; }
    CELEG_TEST_CHECK(caught);

    index.clear();
    CELEG_TEST_CHECK(index.size() == 0 && index.node_count() == 1);
    std::cout << "prefix_radix_test: ok\n";
}
