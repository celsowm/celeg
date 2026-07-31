#pragma once

#include "celeg/detail/runtime/concurrency/request_registry.hpp"

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

// Pure host policy object. It never owns requests and never touches CUDA.
class BatchPlanner {
public:
    std::optional<ConcurrentEngine::RequestId> next_admission(
        const RequestRegistry& registry) const;

    std::vector<int> order_prefill(const std::vector<LaneSnapshot>& lanes) const;
    std::vector<int> order_decode(const std::vector<LaneSnapshot>& lanes) const;

    // Backend-neutral priority/FIFO ordering for request records that are not
    // represented as CUDA lanes (the CPU engine uses this for admission and
    // both batch plans).
    std::vector<size_t> order_priority(
        std::span<const RequestPriorityView> requests) const;
};

} // namespace celeg::detail
