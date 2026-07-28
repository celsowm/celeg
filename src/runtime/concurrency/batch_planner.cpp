#include "lfm/detail/runtime/concurrency/batch_planner.hpp"

#include <algorithm>
#include <numeric>

namespace lfm::detail {

std::optional<ConcurrentEngine::RequestId> BatchPlanner::next_admission(
    const RequestRegistry& registry) const {
    return registry.best_queued();
}

namespace {
std::vector<int> order_for(const std::vector<LaneSnapshot>& lanes,
                           RequestStatus desired) {
    std::vector<int> selected;
    selected.reserve(lanes.size());
    for (size_t i = 0; i < lanes.size(); ++i) {
        const LaneSnapshot& lane = lanes[i];
        if (lane.request_id != 0 && lane.status == desired) {
            selected.push_back(static_cast<int>(i));
        }
    }
    std::sort(selected.begin(), selected.end(),
              [&](int a, int b) {
                  const LaneSnapshot& left = lanes[static_cast<size_t>(a)];
                  const LaneSnapshot& right = lanes[static_cast<size_t>(b)];
                  if (left.priority != right.priority) {
                      return left.priority > right.priority;
                  }
                  return left.request_id < right.request_id;
              });
    std::vector<int> result;
    result.reserve(selected.size());
    for (int index : selected) {
        result.push_back(lanes[static_cast<size_t>(index)].lane_index);
    }
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

std::vector<size_t> BatchPlanner::order_priority(
    std::span<const RequestPriorityView> requests) const {
    std::vector<size_t> result(requests.size());
    std::iota(result.begin(), result.end(), size_t{0});
    std::sort(result.begin(), result.end(), [&](size_t left, size_t right) {
        const auto& a = requests[left];
        const auto& b = requests[right];
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.request_id < b.request_id;
    });
    return result;
}

} // namespace lfm::detail
