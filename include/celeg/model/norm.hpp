#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace celeg {

enum class NormWeightKind : uint8_t {
    Scale,
    OnePlusScale,
    None,
};

/// A present RMS normalization. A norm is valid after construction; absence is
/// modeled by `std::optional<NormSpec>` at the owning position rather than by an
/// invalid (zero-epsilon) norm value.
struct NormSpec {
    float epsilon = 1.0e-5f;
    NormWeightKind weight_kind = NormWeightKind::Scale;

    bool weightless() const { return weight_kind == NormWeightKind::None; }

    void validate() const {
        if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
            throw std::invalid_argument("RMSNorm epsilon must be positive");
        }
    }
};

}