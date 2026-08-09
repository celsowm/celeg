#include "batch_scheduler.hpp"

namespace celeg {
namespace {

std::vector<std::shared_ptr<CpuBatchScheduler::Request>> plan(
    const CpuBatchScheduler::RequestMap& requests,
    const detail::BatchPlanner& planner,
    size_t limit,
    bool prefill) {
    std::vector<std::shared_ptr<CpuBatchScheduler::Request>> candidates;
    for (const auto& [id, request] : requests) {
        (void)id;
        const bool eligible = prefill
            ? request->status == RequestStatus::Prefill &&
              request->prompt_offset < request->prompt.size()
            : request->status == RequestStatus::Decoding;
        if (eligible) candidates.push_back(request);
    }
    std::vector<detail::RequestPriorityView> views;
    views.reserve(candidates.size());
    for (const auto& request : candidates) {
        views.push_back({request->id, request->options.priority});
    }
    const auto order = planner.order_priority(views);
    std::vector<std::shared_ptr<CpuBatchScheduler::Request>> result;
    result.reserve(order.size());
    for (const size_t index : order) result.push_back(candidates[index]);
    if (result.size() > limit) result.resize(limit);
    return result;
}

} // namespace

std::vector<std::shared_ptr<CpuBatchScheduler::Request>> CpuBatchScheduler::plan_decode(
    const RequestMap& requests, const detail::BatchPlanner& planner, size_t limit) {
    return plan(requests, planner, limit, false);
}

std::vector<std::shared_ptr<CpuBatchScheduler::Request>> CpuBatchScheduler::plan_prefill(
    const RequestMap& requests, const detail::BatchPlanner& planner, size_t limit) {
    return plan(requests, planner, limit, true);
}

} // namespace celeg
