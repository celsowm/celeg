#pragma once

#include "celeg/runtime/request_types.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

class CpuModel;

struct CpuConcurrentRequest {
    RequestId id = 0;
    RequestStatus status = RequestStatus::Queued;
    std::vector<int32_t> prompt;
    size_t prompt_offset = 0;
    ConcurrentRequestOptions options;
    std::unique_ptr<CpuModel> session;
    bool cancel_requested = false;
    std::vector<int32_t> generated;
    size_t poll_offset = 0;
    uint64_t sequence = 0;
    std::chrono::steady_clock::time_point submitted_at;
    std::chrono::steady_clock::time_point last_token_at;
    bool first_token_recorded = false;
    bool prefix_inserted = false;
    uint64_t attention_parallel_observed = 0;
    int numa_node = -1;
    std::string error;
};

}
