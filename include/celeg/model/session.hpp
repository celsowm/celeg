#pragma once

#include "celeg/model/runtime_types.hpp"

#include <utility>

namespace celeg {

// Request-local state. It is deliberately backend-neutral; KV and device
// buffers are owned by the selected backend and referenced by operation
// contexts rather than embedded in this type.
struct SessionState {
    explicit SessionState(GenerationConfig generation_config = {})
        : generation_(std::move(generation_config)) {}

    GenerationConfig generation_;
    int position_ = 0;
    SessionPhase phase_ = SessionPhase::Empty;
    bool active_segmented_attention_ = false;
    RuntimeMetrics metrics_;
};

} // namespace celeg
