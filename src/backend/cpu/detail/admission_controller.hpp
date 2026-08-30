#pragma once

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/backend/cpu/numa_placement.hpp"
#include "celeg/backend/cpu/prefix_cache.hpp"
#include "runtime/concurrency/detail/batch_planner.hpp"
#include "celeg/runtime/concurrency/metrics.hpp"
#include "concurrent_request.hpp"

#include <memory>
#include <unordered_map>

namespace celeg {

class CpuAdmissionController {
public:
    using Request = CpuConcurrentRequest;
    using RequestMap = std::unordered_map<RequestId, std::shared_ptr<Request>>;

    CpuAdmissionController(
        CpuModel& base_model,
        CpuNumaPlacement& numa_placement,
        std::unique_ptr<CpuPrefixCacheManager>& prefix_cache)
        : base_model_(base_model), numa_placement_(numa_placement),
          prefix_cache_(prefix_cache) {}

    void admit(RequestMap& requests,
               const detail::BatchPlanner& planner,
               size_t maximum_active_requests,
               ConcurrentMetrics& metrics);
    void sync_prefix_metrics(ConcurrentMetrics& metrics) const;

private:
    CpuModel& base_model_;
    CpuNumaPlacement& numa_placement_;
    std::unique_ptr<CpuPrefixCacheManager>& prefix_cache_;
};

}
