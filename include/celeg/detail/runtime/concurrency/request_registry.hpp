#pragma once

#include "celeg/backend/cuda/concurrency.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace celeg::detail {

struct RequestRecord {
    ConcurrentEngine::RequestId id = 0;
    ConcurrentRequestOptions options;
    std::vector<int32_t> prompt;
    size_t prefill_offset = 0;
    int generated = 0;
    RequestStatus status = RequestStatus::Queued;
    std::deque<int32_t> output;
    std::vector<uint32_t> pages;
    int lane_index = -1;
    bool cancel_requested = false;
    bool paged_ready = false;
    std::string error;
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point last_token_at{};
};

class RequestRegistry {
public:
    using RequestId = ConcurrentEngine::RequestId;

    RequestId create(std::vector<int32_t> prompt,
                     ConcurrentRequestOptions options,
                     std::chrono::steady_clock::time_point submitted_at);

    RequestRecord* find(RequestId id);
    const RequestRecord* find(RequestId id) const;
    RequestRecord& at(RequestId id);
    const RequestRecord& at(RequestId id) const;

    bool empty_queue() const { return admission_queue_.empty(); }
    size_t queued_count() const { return admission_queue_.size(); }
    // Returns the highest-priority (best) queued request, or nullopt.
    std::optional<RequestId> best_queued() const;
    void erase_queued(RequestId id);

    // Fully removes the request record. Returns false if unknown.
    bool erase(RequestId id);

private:
    std::unordered_map<RequestId, std::unique_ptr<RequestRecord>> requests_;
    // Ordered set: highest priority first, then lowest ID first (FIFO).
    // Key = (-priority, id).  O(log Q) insert/erase/best.
    std::set<std::pair<int, RequestId>> admission_queue_;
    RequestId next_request_id_ = 1;
};

} // namespace celeg::detail
