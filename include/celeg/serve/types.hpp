#pragma once

#include "celeg/model/execution/runtime_types.hpp"
#include "celeg/runtime/concurrency.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace celeg::serve {

using RequestId = celeg::RequestId;

// Backend-neutral request lifecycle shared by both inference backends.
using RequestStatus = celeg::RequestStatus;

enum class FinishReason : std::uint8_t {
    None,
    Stop,
    Length,
    Cancelled,
    Error,
};

struct GenerateRequest {
    std::vector<std::int32_t> prompt_tokens;
    GenerationConfig generation;
    std::size_t max_output_tokens = 128;
    std::int32_t eos_token_id = 7;
    int priority = 0;
};

struct GenerateEvent {
    RequestId request_id = 0;
    std::vector<std::int32_t> tokens;
    RequestStatus status = RequestStatus::Queued;
    bool finished = false;
    FinishReason finish_reason = FinishReason::None;
    std::string error;
};

struct ModelInfo {
    std::string name;
    std::string backend;
    int max_context = 0;
};

struct ServingMetrics {
    std::uint64_t submitted_requests = 0;
    std::uint64_t completed_requests = 0;
    std::uint64_t cancelled_requests = 0;
    std::uint64_t failed_requests = 0;
    std::uint64_t active_requests = 0;
    std::uint64_t queued_requests = 0;
    double prefill_tokens_per_second = 0.0;
    double decode_tokens_per_second = 0.0;
    double average_ttft_ms = 0.0;
    double average_itl_ms = 0.0;
};

} // namespace celeg::serve
