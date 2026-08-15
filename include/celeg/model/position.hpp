#pragma once

#include "celeg/model/definition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace celeg {

inline double rope_frequency(const RopePositionSpec& spec, int pair,
                             int rotary_dimension, int position) {
    if (pair < 0 || pair >= rotary_dimension / 2 || rotary_dimension <= 0 ||
        (rotary_dimension % 2) != 0 || position < 0) {
        throw std::invalid_argument("invalid RoPE frequency coordinates");
    }
    const double base = spec.theta;
    const double context = static_cast<double>(position);
    const auto scaled = [&](double base_frequency) -> double {
        const double frequency = std::visit([&](const auto& scaling) -> double {
            using Scaling = std::decay_t<decltype(scaling)>;
            double result = base_frequency;
            if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
                return result;
            } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
                result /= scaling.factor;
                return result;
            } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
                return result;
            } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
                const double ramp = std::clamp(
                    (static_cast<double>(pair) - scaling.beta_fast) /
                        std::max(1.0, scaling.beta_slow - scaling.beta_fast),
                    0.0, 1.0);
                result /= 1.0 + ramp * (scaling.factor - 1.0);
                return result;
            } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
                const auto& factors = context > static_cast<double>(scaling.original_context)
                    ? scaling.long_factors : scaling.short_factors;
                result /= static_cast<double>(factors[static_cast<size_t>(pair)]);
                return result;
            } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
                const double wavelength = 2.0 * 3.14159265358979323846 / result;
                if (wavelength < static_cast<double>(scaling.original_context) /
                                 scaling.high_frequency_factor) {
                    return result;
                }
                if (wavelength > static_cast<double>(scaling.original_context) /
                                 scaling.low_frequency_factor) {
                    result /= scaling.factor;
                    return result;
                }
                const double span = scaling.low_frequency_factor - scaling.high_frequency_factor;
                const double blend = span > 0.0
                    ? (wavelength * scaling.high_frequency_factor /
                       static_cast<double>(scaling.original_context) - 1.0) / span
                    : 0.0;
                result /= 1.0 + std::clamp(blend, 0.0, 1.0) * (scaling.factor - 1.0);
                return result;
            } else {
                static_assert(always_false_v<Scaling>, "unhandled RoPE scaling variant");
            }
        }, spec.scaling);
        return frequency;
    };
    double frequency = scaled(std::pow(base, -2.0 * static_cast<double>(pair) /
                                           static_cast<double>(rotary_dimension)));
    if (const auto* ntk = std::get_if<DynamicNtkRopeScaling>(&spec.scaling);
        ntk && context > static_cast<double>(ntk->original_context)) {
        const double ratio = ntk->factor * context /
            static_cast<double>(ntk->original_context) - (ntk->factor - 1.0);
        frequency *= std::pow(std::max(1.0, ratio),
                              static_cast<double>(rotary_dimension) /
                                  static_cast<double>(std::max(2, rotary_dimension - 2)));
    }
    return frequency;
}

inline float rope_attention_scale(const RopePositionSpec& spec, int position) {
    if (position <= 0) return 1.0f;
    if (const auto* yarn = std::get_if<YarnRopeScaling>(&spec.scaling)) {
        return static_cast<float>(yarn->attention_factor);
    }
    return 1.0f;
}

}
