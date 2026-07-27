#pragma once

#include "lfm/detail/runtime/concurrency/request_registry.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace lfm::detail {

struct LaneSnapshot {
    int lane_index = -1;
    ConcurrentEngine::RequestId request_id = 0;
    RequestStatus status = RequestStatus::Queued;
    int priority = 0;
};

// Pure host policy object. It never owns requests and never touches CUDA.
class BatchPlanner {
public:
    std::optional<ConcurrentEngine::RequestId> next_admission(
        const RequestRegistry& registry) const;

    std::vector<int> order_prefill(const std::vector<LaneSnapshot>& lanes) const;
    std::vector<int> order_decode(const std::vector<LaneSnapshot>& lanes) const;
};

} // namespace lfm::detail
