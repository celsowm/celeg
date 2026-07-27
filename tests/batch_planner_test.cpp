#include "lfm/detail/runtime/concurrency/batch_planner.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    lfm::detail::RequestRegistry registry;
    lfm::ConcurrentRequestOptions low;
    low.priority = 1;
    lfm::ConcurrentRequestOptions high;
    high.priority = 7;
    const auto now = std::chrono::steady_clock::now();
    const auto first = registry.create({1}, low, now);
    const auto second = registry.create({2}, high, now);

    lfm::detail::BatchPlanner planner;
    assert(planner.next_admission(registry).value() == second);

    std::vector<lfm::detail::LaneSnapshot> lanes = {
        {0, first, lfm::RequestStatus::Prefill, 1},
        {1, second, lfm::RequestStatus::Prefill, 7},
        {2, 42, lfm::RequestStatus::Decoding, 3},
        {3, 43, lfm::RequestStatus::Decoding, 8},
    };
    const auto prefill = planner.order_prefill(lanes);
    assert((prefill == std::vector<int>{1, 0}));
    const auto decode = planner.order_decode(lanes);
    assert((decode == std::vector<int>{3, 2}));
    std::cout << "batch_planner_test: ok\n";
}
