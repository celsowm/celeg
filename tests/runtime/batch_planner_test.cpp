#include "celeg/detail/runtime/concurrency/batch_planner.hpp"
#include "support/assertions.hpp"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    celeg::detail::RequestRegistry registry;
    celeg::ConcurrentRequestOptions low;
    low.priority = 1;
    celeg::ConcurrentRequestOptions high;
    high.priority = 7;
    const auto now = std::chrono::steady_clock::now();
    const auto first = registry.create({1}, low, now);
    const auto second = registry.create({2}, high, now);

    celeg::detail::BatchPlanner planner;
    CELEG_TEST_CHECK(planner.next_admission(registry).value() == second);

    std::vector<celeg::detail::LaneSnapshot> lanes = {
        {0, first, celeg::RequestStatus::Prefill, 1},
        {1, second, celeg::RequestStatus::Prefill, 7},
        {2, 42, celeg::RequestStatus::Decoding, 3},
        {3, 43, celeg::RequestStatus::Decoding, 8},
    };
    const auto prefill = planner.order_prefill(lanes);
    CELEG_TEST_CHECK((prefill == std::vector<int>{1, 0}));
    const auto decode = planner.order_decode(lanes);
    CELEG_TEST_CHECK((decode == std::vector<int>{3, 2}));
    const std::vector<celeg::detail::RequestPriorityView> requests = {
        {10, 1}, {11, 7}, {12, 7},
    };
    CELEG_TEST_CHECK((planner.order_priority(requests) ==
                    std::vector<size_t>{1, 2, 0}));
    std::cout << "batch_planner_test: ok\n";
}
