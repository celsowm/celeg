#include "admission_controller.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace celeg {

void CpuAdmissionController::sync_prefix_metrics(
    ConcurrentMetrics& metrics) const {
    if (!prefix_cache_) return;
    const auto& source = prefix_cache_->metrics();
    metrics.prefix_cache_hits = source.hits;
    metrics.prefix_cache_misses = source.misses;
    metrics.prefix_cache_partial_hits = source.partial_hits;
    metrics.prefix_cache_inserts = source.inserts;
    metrics.prefix_cache_evictions = source.evictions;
    metrics.prefix_reused_tokens = source.reused_tokens;
    metrics.prefix_cow_pages = source.cow_pages;
    metrics.prefix_cow_bytes_copied = source.cow_bytes;
    metrics.prefix_radix_lookups = source.radix_lookups;
    metrics.prefix_radix_nodes = prefix_cache_->radix_nodes();
}

void CpuAdmissionController::admit(
    RequestMap& requests,
    const detail::BatchPlanner& planner,
    size_t maximum_active_requests,
    ConcurrentMetrics& metrics) {
    size_t active = 0;
    std::vector<std::shared_ptr<Request>> queued;
    for (const auto& [id, request] : requests) {
        (void)id;
        if (request->status == RequestStatus::Prefill ||
            request->status == RequestStatus::Decoding) {
            ++active;
        } else if (request->status == RequestStatus::Queued) {
            queued.push_back(request);
        }
    }
    const size_t available = maximum_active_requests -
        std::min(maximum_active_requests, active);
    if (available == 0 || queued.empty()) return;

    std::vector<detail::RequestPriorityView> queued_views;
    queued_views.reserve(queued.size());
    for (const auto& request : queued) {
        queued_views.push_back({request->id, request->options.priority});
    }
    const auto queued_order = planner.order_priority(queued_views);
    std::vector<std::shared_ptr<Request>> ordered_queued;
    ordered_queued.reserve(queued.size());
    for (const size_t index : queued_order) ordered_queued.push_back(queued[index]);

    size_t remaining = available;
    for (const auto& request : ordered_queued) {
        if (remaining == 0) break;
        try {
            request->numa_node = numa_placement_.next_node();
            request->session = base_model_.clone_session_on_node(request->numa_node);
            request->session->session().set_generation_config(request->options.generation);
            request->status = RequestStatus::Prefill;
            if (prefix_cache_ && request->options.prompt_embedding.empty()) {
                if (auto match = prefix_cache_->acquire(
                        request->prompt, request->numa_node)) {
                    const bool exact = match->matched_tokens == request->prompt.size();
                    request->session->persistence().restore_prefix_snapshot(
                        std::move(match->snapshot), exact);
                    request->prompt_offset = match->matched_tokens;
                    request->status = exact ? RequestStatus::Decoding
                                            : RequestStatus::Prefill;
                }
                sync_prefix_metrics(metrics);
            }
            --remaining;
        } catch (const std::exception& error) {
            request->status = RequestStatus::Failed;
            request->error = error.what();
            ++metrics.failed;
        }
    }
}

}
