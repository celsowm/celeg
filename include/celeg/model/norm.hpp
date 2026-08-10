#pragma once

#include <cstdint>
#include <stdexcept>

namespace celeg {

// Mathematical convention used by an RMSNorm weight.  This is deliberately
// independent of checkpoint spelling and backend implementation.
enum class NormWeightKind : uint8_t {
    Scale,
    OnePlusScale,
    None,
};

struct NormSpec {
    float epsilon = 0.0f;
    NormWeightKind weight_kind = NormWeightKind::Scale;

    bool weightless() const { return weight_kind == NormWeightKind::None; }
    bool enabled() const { return epsilon > 0.0f; }
    void validate() const {
        if (!(epsilon > 0.0f)) {
            throw std::invalid_argument("RMSNorm epsilon must be positive");
        }
    }
};

} // namespace celeg
