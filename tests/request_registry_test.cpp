#include "lfm/detail/runtime/concurrency/request_registry.hpp"

#include <cassert>
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

    assert(first != second);
    assert(registry.queued_count() == 2);
    assert(registry.at(first).prompt.size() == 2);
    assert(registry.find(second)->options.priority == 9);

    registry.erase_queued(first);
    assert(registry.queued_count() == 1);
    assert(registry.best_queued().has_value());
    assert(*registry.best_queued() == second);
    assert(registry.find(999999) == nullptr);

    bool threw = false;
    try {
        (void)registry.at(999999);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    std::cout << "request_registry_test: ok\n";
}
