#include "runtime/concurrency/detail/request_registry.hpp"
#include "support/assertions.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    celeg::detail::RequestRegistry registry;
    celeg::ConcurrentRequestOptions low;
    low.priority = 1;
    celeg::ConcurrentRequestOptions high;
    high.priority = 9;
    const auto now = std::chrono::steady_clock::now();
    const auto first = registry.create({1, 2}, low, now);
    const auto second = registry.create({3}, high, now);

    CELEG_TEST_CHECK(first != second);
    CELEG_TEST_CHECK(registry.queued_count() == 2);
    CELEG_TEST_CHECK(registry.at(first).prompt.size() == 2);
    CELEG_TEST_CHECK(registry.find(second)->options.priority == 9);

    registry.erase_queued(first);
    CELEG_TEST_CHECK(registry.queued_count() == 1);
    CELEG_TEST_CHECK(registry.best_queued().has_value());
    CELEG_TEST_CHECK(*registry.best_queued() == second);
    CELEG_TEST_CHECK(registry.find(999999) == nullptr);

    bool threw = false;
    try {
        (void)registry.at(999999);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);

    CELEG_TEST_CHECK(registry.erase(second) == true);
    CELEG_TEST_CHECK(registry.find(second) == nullptr);
    CELEG_TEST_CHECK(registry.queued_count() == 0);
    CELEG_TEST_CHECK(registry.erase(second) == false);
    CELEG_TEST_CHECK(registry.erase(999999) == false);

    std::cout << "request_registry_test: ok\n";
}
