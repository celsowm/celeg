#pragma once

#include "celeg/serve/types.hpp"

#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>

namespace celeg::serve {

// Backend-neutral request lifecycle state. Engines only produce tokens and
// statuses; this component owns EOS observation and the public finish reason.
class RequestLifecycle final {
public:
    void submitted(RequestId id, std::int32_t eos_token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_[id] = State{eos_token_id, false};
    }

    FinishReason finish_reason(RequestId id, RequestStatus status,
                               std::span<const std::int32_t> tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = states_.find(id);
        if (it != states_.end() && !tokens.empty() &&
            tokens.back() == it->second.eos_token_id) {
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
        std::int32_t eos_token_id;
        bool saw_eos;
    };

    mutable std::mutex mutex_;
    std::unordered_map<RequestId, State> states_;
};

} // namespace celeg::serve
