#pragma once

#include "celeg/model/definition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {

// Canonical frequency transform shared by position-aware lowerings.  The
// position argument is deliberately part of the function: dynamic NTK and
// long-context policies are position-dependent, not merely alternate bases.
inline double rope_frequency(const RopePositionSpec& spec, int pair,
                             int rotary_dimension, int position) {
    if (pair < 0 || pair >= rotary_dimension / 2 || rotary_dimension <= 0 ||
        (rotary_dimension % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid RoPE frequency coordinates");
    }
    const RopeScalingSpec& scaling = spec.scaling;
    double base = spec.theta;
    const double context = static_cast<double>(position);
    if (scaling.kind == RopeScalingKind::DynamicNtk &&
        context > static_cast<double>(scaling.original_context)) {
        const double ratio = scaling.factor * context /
            static_cast<double>(scaling.original_context) - (scaling.factor - 1.0);
        base *= std::pow(std::max(1.0, ratio),
                         static_cast<double>(rotary_dimension) /
                             static_cast<double>(std::max(2, rotary_dimension - 2)));
    }
    double frequency = std::pow(base, -2.0 * static_cast<double>(pair) /
                                      static_cast<double>(rotary_dimension));
    switch (scaling.kind) {
    case RopeScalingKind::None:
        break;
    case RopeScalingKind::Linear:
        frequency /= scaling.factor;
        break;
    case RopeScalingKind::DynamicNtk:
        break;
    case RopeScalingKind::Yarn: {
        const double ramp = std::clamp(
            (static_cast<double>(pair) - scaling.beta_fast) /
                std::max(1.0, scaling.beta_slow - scaling.beta_fast),
            0.0, 1.0);
        frequency /= 1.0 + ramp * (scaling.factor - 1.0);
        break;
    }
    case RopeScalingKind::Long: {
        const auto& factors = context > static_cast<double>(scaling.original_context)
            ? scaling.long_factors : scaling.short_factors;
        frequency /= static_cast<double>(factors[static_cast<size_t>(pair)]);
        break;
    }
    case RopeScalingKind::Llama3Frequency: {
        const double wavelength = 2.0 * 3.14159265358979323846 / frequency;
        if (wavelength < static_cast<double>(scaling.original_context) /
                         scaling.high_frequency_factor) {
            break;
        }
        if (wavelength > static_cast<double>(scaling.original_context) /
                         scaling.low_frequency_factor) {
            frequency /= scaling.factor;
            break;
        }
        const double span = scaling.low_frequency_factor - scaling.high_frequency_factor;
        const double blend = span > 0.0
            ? (wavelength * scaling.high_frequency_factor /
               static_cast<double>(scaling.original_context) - 1.0) / span
            : 0.0;
        frequency /= 1.0 + std::clamp(blend, 0.0, 1.0) * (scaling.factor - 1.0);
        break;
    }
    }
    return frequency;
}

inline float rope_attention_scale(const RopePositionSpec& spec, int position) {
    if (spec.scaling.kind != RopeScalingKind::Yarn || position <= 0) return 1.0f;
    return static_cast<float>(spec.scaling.attention_factor);
}

} // namespace celeg
