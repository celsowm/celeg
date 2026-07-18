#include "lfm/prefix_radix.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    lfm::PrefixRadixIndex index;
    assert(index.size() == 0);
    assert(index.node_count() == 1);
    assert(!index.longest_prefix({1, 2, 3}));

    index.insert({1, 2}, 10);
    index.insert({1, 2, 3, 4}, 20);
    index.insert({1, 5}, 30);
    assert(index.size() == 3);

    auto match = index.longest_prefix({1, 2, 3, 4, 9});
    assert(match && match->id == 20 && match->matched_tokens == 4);
    match = index.longest_prefix({1, 2, 8});
    assert(match && match->id == 10 && match->matched_tokens == 2);
    match = index.longest_prefix({1, 5, 7});
    assert(match && match->id == 30 && match->matched_tokens == 2);
    assert(!index.longest_prefix({9, 9}));

    assert(index.erase({1, 2, 3, 4}, 20));
    match = index.longest_prefix({1, 2, 3, 4, 9});
    assert(match && match->id == 10 && match->matched_tokens == 2);
    assert(!index.erase({1, 2, 3, 4}, 20));
    assert(index.erase({1, 2}, 10));
    assert(!index.longest_prefix({1, 2, 8}));
    assert(index.size() == 1);

    bool caught = false;
    try { index.insert({}, 99); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);
    caught = false;
    try { index.insert({1}, 0); } catch (const std::invalid_argument&) { caught = true; }
    assert(caught);

    index.clear();
    assert(index.size() == 0 && index.node_count() == 1);
    std::cout << "prefix_radix_test: ok\n";
}
