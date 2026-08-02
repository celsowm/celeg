#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/runtime_types.hpp"

#include <cstdint>
#include <span>

namespace celeg {

class CpuSampler final {
public:
    static std::int32_t sample(std::span<const float> logits,
                               const RuntimeTopology& shape,
                               const GenerationConfig& generation,
                               std::span<const std::uint8_t> seen,
                               std::uint64_t& rng_state);
};

} // namespace celeg
