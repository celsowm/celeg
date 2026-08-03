#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy.hpp"

namespace celeg {

using RequestId = std::uint64_t;

struct ConcurrentRequestOptions {
    int max_new_tokens = 128;
    int eos_token = 7;
    int priority = 0;
    GenerationConfig generation{};
    PromptEmbedding prompt_embedding;
};

struct PollResult {
    RequestStatus status = RequestStatus::Queued;
    std::vector<int32_t> tokens;
    bool finished = false;
    std::string error;
};

} // namespace celeg
