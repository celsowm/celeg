#include "detail/batch_scheduler.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>

int main() {
    celeg::CpuBatchScheduler::RequestMap requests;
    auto low = std::make_shared<celeg::CpuConcurrentRequest>();
    low->id = 1;
    low->status = celeg::RequestStatus::Decoding;
    low->options.priority = 1;
    auto high = std::make_shared<celeg::CpuConcurrentRequest>();
    high->id = 2;
    high->status = celeg::RequestStatus::Decoding;
    high->options.priority = 7;
    requests.emplace(low->id, low);
    requests.emplace(high->id, high);

    celeg::detail::BatchPlanner planner;
    const auto plan = celeg::CpuBatchScheduler::plan_decode(requests, planner, 2);
    CELEG_TEST_CHECK(plan.size() == 2);
    CELEG_TEST_CHECK(plan[0]->id == high->id);
    CELEG_TEST_CHECK(plan[1]->id == low->id);
    CELEG_TEST_CHECK(low->status == celeg::RequestStatus::Decoding);
    std::cout << "cpu_batch_scheduler_test: ok\n";
}
