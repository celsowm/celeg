#include "lfm/detail/batch_planner.hpp"

#include <algorithm>

namespace lfm::detail {

std::optional<ConcurrentEngine::RequestId> BatchPlanner::next_admission(
    const RequestRegistry& registry) const {
    return registry.best_queued();
}

namespace {
std::vector<int> order_for(const std::vector<LaneSnapshot>& lanes,
                           RequestStatus desired) {
    std::vector<LaneSnapshot> selected;
    selected.reserve(lanes.size());
    for (const LaneSnapshot& lane : lanes) {
        if (lane.request_id != 0 && lane.status == desired) selected.push_back(lane);
    }
    std::stable_sort(selected.begin(), selected.end(),
                     [](const LaneSnapshot& a, const LaneSnapshot& b) {
                         if (a.priority != b.priority) return a.priority > b.priority;
                         return a.request_id < b.request_id;
                     });
    std::vector<int> result;
    result.reserve(selected.size());
    for (const LaneSnapshot& lane : selected) result.push_back(lane.lane_index);
    return result;
}
} // namespace

std::vector<int> BatchPlanner::order_prefill(
    const std::vector<LaneSnapshot>& lanes) const {
    return order_for(lanes, RequestStatus::Prefill);
}

std::vector<int> BatchPlanner::order_decode(
    const std::vector<LaneSnapshot>& lanes) const {
    return order_for(lanes, RequestStatus::Decoding);
}

} // namespace lfm::detail
