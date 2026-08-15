#include "celeg/runtime/concurrency/policy.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>

int main() {

    celeg::PagedBlockPool pool(8, 16);
    auto empty = pool.allocate_tokens(0);
    CELEG_TEST_CHECK(empty && empty->empty());
    auto a = pool.allocate_tokens(1);
    CELEG_TEST_CHECK(a && a->size() == 1);
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == 1);
    auto b = pool.allocate_tokens(33);
    CELEG_TEST_CHECK(b && b->size() == 3);
    CELEG_TEST_CHECK(pool.used_pages() == 4);
    const bool retained_a = pool.retain(*a);
    CELEG_TEST_CHECK(retained_a);
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == 2);
    pool.release(*a);
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == 1);
    CELEG_TEST_CHECK(pool.used_pages() == 4);
    const bool retained_invalid = pool.retain(std::vector<uint32_t>{99});
    CELEG_TEST_CHECK(!retained_invalid);

    const bool retained_duplicates = pool.retain(std::vector<uint32_t>{a->front(), a->front()});
    CELEG_TEST_CHECK(retained_duplicates);
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == 3);
    pool.release(std::vector<uint32_t>{a->front(), a->front()});
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == 1);

    const uint32_t before = pool.ref_count(a->front());
    bool transactional_caught = false;
    try {
        pool.release(std::vector<uint32_t>{a->front(), 99});
    } catch (const std::runtime_error&) {
        transactional_caught = true;
    }
    CELEG_TEST_CHECK(transactional_caught);
    CELEG_TEST_CHECK(pool.ref_count(a->front()) == before);

    auto too_large = pool.allocate_tokens(80);
    CELEG_TEST_CHECK(!too_large);
    pool.release(*a);
    pool.release(*b);
    CELEG_TEST_CHECK(pool.free_pages() == 8);
    bool caught = false;
    try { pool.release(*a); } catch (const std::runtime_error&) { caught = true; }
    CELEG_TEST_CHECK(caught);
    std::cout << "concurrent_policy_test: ok\n";
}
