#pragma once

#include "celeg/serve/types.hpp"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>

namespace celeg::serve {

class RequestLifecycle final {
public:
    void submitted(RequestId id, std::span<const std::int32_t> eos_token_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_[id] = State{std::vector<std::int32_t>(eos_token_ids.begin(), eos_token_ids.end()), false};
    }

    FinishReason finish_reason(RequestId id, RequestStatus status,
                               std::span<const std::int32_t> tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = states_.find(id);
        if (it != states_.end() && !tokens.empty() &&
            std::find(it->second.eos_token_ids.begin(), it->second.eos_token_ids.end(),
                      tokens.back()) != it->second.eos_token_ids.end()) {
            it->second.saw_eos = true;
        }
        const bool saw_eos = it != states_.end() && it->second.saw_eos;
        switch (status) {
            case RequestStatus::Cancelled: return FinishReason::Cancelled;
            case RequestStatus::Failed: return FinishReason::Error;
            case RequestStatus::Finished:
                return saw_eos ? FinishReason::Stop : FinishReason::Length;
            default: return FinishReason::None;
        }
    }

    void released(RequestId id) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.erase(id);
    }

private:
    struct State {
        std::vector<std::int32_t> eos_token_ids;
        bool saw_eos;
    };

    mutable std::mutex mutex_;
    std::unordered_map<RequestId, State> states_;
};

}
