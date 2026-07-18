#include "lfm/concurrent_policy.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {

    lfm::PagedBlockPool pool(8, 16);
    auto empty = pool.allocate_tokens(0);
    assert(empty && empty->empty());
    auto a = pool.allocate_tokens(1);
    assert(a && a->size() == 1);
    assert(pool.ref_count(a->front()) == 1);
    auto b = pool.allocate_tokens(33);
    assert(b && b->size() == 3);
    assert(pool.used_pages() == 4);
    const bool retained_a = pool.retain(*a);
    assert(retained_a);
    assert(pool.ref_count(a->front()) == 2);
    pool.release(*a);
    assert(pool.ref_count(a->front()) == 1);
    assert(pool.used_pages() == 4);
    const bool retained_invalid = pool.retain(std::vector<uint32_t>{99});
    assert(!retained_invalid);

    // Duplicate IDs are counted transactionally.
    const bool retained_duplicates = pool.retain(std::vector<uint32_t>{a->front(), a->front()});
    assert(retained_duplicates);
    assert(pool.ref_count(a->front()) == 3);
    pool.release(std::vector<uint32_t>{a->front(), a->front()});
    assert(pool.ref_count(a->front()) == 1);

    // An invalid item later in a release list must not mutate earlier pages.
    const uint32_t before = pool.ref_count(a->front());
    bool transactional_caught = false;
    try {
        pool.release(std::vector<uint32_t>{a->front(), 99});
    } catch (const std::runtime_error&) {
        transactional_caught = true;
    }
    assert(transactional_caught);
    assert(pool.ref_count(a->front()) == before);

    auto too_large = pool.allocate_tokens(80);
    assert(!too_large);
    pool.release(*a);
    pool.release(*b);
    assert(pool.free_pages() == 8);
    bool caught = false;
    try { pool.release(*a); } catch (const std::runtime_error&) { caught = true; }
    assert(caught);
    std::cout << "concurrent_policy_test: ok\n";
}
