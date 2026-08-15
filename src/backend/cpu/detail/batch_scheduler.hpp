#pragma once

#include "celeg/detail/runtime/concurrency/batch_planner.hpp"
#include "concurrent_request.hpp"

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

namespace celeg {

class CpuBatchScheduler {
public:
    using Request = CpuConcurrentRequest;
    using RequestMap = std::unordered_map<RequestId, std::shared_ptr<Request>>;

    static std::vector<std::shared_ptr<Request>> plan_decode(
        const RequestMap& requests, const detail::BatchPlanner& planner, size_t limit);
    static std::vector<std::shared_ptr<Request>> plan_prefill(
        const RequestMap& requests, const detail::BatchPlanner& planner, size_t limit);
};

}
