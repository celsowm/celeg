#pragma once

#include "celeg/model/runtime_types.hpp"

#include <utility>
#include <array>

namespace celeg {

struct SessionState {
    explicit SessionState(GenerationConfig generation_config = {})
        : generation_(std::move(generation_config)) {}

    GenerationConfig generation_;
    int position_ = 0;
    std::array<int32_t, 3> next_rope_position_{0, 0, 0};
    SessionPhase phase_ = SessionPhase::Empty;
    bool active_segmented_attention_ = false;
    RuntimeMetrics metrics_;
};

}
