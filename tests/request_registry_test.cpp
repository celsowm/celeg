#include "lfm/detail/runtime/concurrency/request_registry.hpp"
#include "support/assertions.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    lfm::detail::RequestRegistry registry;
    lfm::ConcurrentRequestOptions low;
    low.priority = 1;
    lfm::ConcurrentRequestOptions high;
    high.priority = 9;
    const auto now = std::chrono::steady_clock::now();
    const auto first = registry.create({1, 2}, low, now);
    const auto second = registry.create({3}, high, now);

    LFM_TEST_CHECK(first != second);
    LFM_TEST_CHECK(registry.queued_count() == 2);
    LFM_TEST_CHECK(registry.at(first).prompt.size() == 2);
    LFM_TEST_CHECK(registry.find(second)->options.priority == 9);

    registry.erase_queued(first);
    LFM_TEST_CHECK(registry.queued_count() == 1);
    LFM_TEST_CHECK(registry.best_queued().has_value());
    LFM_TEST_CHECK(*registry.best_queued() == second);
    LFM_TEST_CHECK(registry.find(999999) == nullptr);

    bool threw = false;
    try {
        (void)registry.at(999999);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    LFM_TEST_CHECK(threw);

    LFM_TEST_CHECK(registry.erase(second) == true);
    LFM_TEST_CHECK(registry.find(second) == nullptr);
    LFM_TEST_CHECK(registry.queued_count() == 0);
    LFM_TEST_CHECK(registry.erase(second) == false);
    LFM_TEST_CHECK(registry.erase(999999) == false);

    std::cout << "request_registry_test: ok\n";
}
