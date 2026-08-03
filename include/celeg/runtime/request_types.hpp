#pragma once

#include <cstdint>
#include <algorithm>
#include <span>
#include <string>
#include <vector>

#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy.hpp"

namespace celeg {

using RequestId = std::uint64_t;

struct ConcurrentRequestOptions {
    int max_new_tokens = 128;
    std::vector<int> eos_tokens = {7};
    int priority = 0;
    GenerationConfig generation{};
    PromptEmbedding prompt_embedding;
};

inline bool is_stop_token(std::span<const int> stop_tokens, int token) {
    return std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end();
}

struct PollResult {
    RequestStatus status = RequestStatus::Queued;
    std::vector<int32_t> tokens;
    bool finished = false;
    std::string error;
};

} // namespace celeg
