#pragma once

#include "request_registry.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace celeg::detail {

struct LaneSnapshot {
    int lane_index = -1;
    ConcurrentEngine::RequestId request_id = 0;
    RequestStatus status = RequestStatus::Queued;
    int priority = 0;
};

struct RequestPriorityView {
    uint64_t request_id = 0;
    int priority = 0;
};

class BatchPlanner {
public:
    std::optional<ConcurrentEngine::RequestId> next_admission(
        const RequestRegistry& registry) const;

    std::vector<int> order_prefill(const std::vector<LaneSnapshot>& lanes) const;
    std::vector<int> order_decode(const std::vector<LaneSnapshot>& lanes) const;

    std::vector<size_t> order_priority(
        std::span<const RequestPriorityView> requests) const;
};

}
