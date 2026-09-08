#pragma once

#include "kernels/rope.hpp"

namespace celeg::cuda_rope {

__device__ __forceinline__ float yarn_correction_dimension(
    float rotations, int rotary_dimension, float theta, int original_context) {
    const float numerator = static_cast<float>(rotary_dimension) *
        logf(static_cast<float>(original_context) /
             (rotations * 6.28318530717958647692f));
    const float denominator = 2.0f * logf(theta);
    return numerator / denominator;
}

__device__ __forceinline__ float scaled_frequency(
    float theta, int pair, int rotary_dimension, int position,
    CudaRopeScaling scaling) {
    float base = theta;
    if (scaling.kind == 2 && scaling.original_context > 0 &&
        position > scaling.original_context) {
        const float context = static_cast<float>(position);
        const float ratio = scaling.factor * context /
            static_cast<float>(scaling.original_context) - (scaling.factor - 1.0f);
        const int denominator = rotary_dimension - 2 > 2 ? rotary_dimension - 2 : 2;
        base *= powf(fmaxf(1.0f, ratio),
                     static_cast<float>(rotary_dimension) /
                     static_cast<float>(denominator));
    }
    float frequency = powf(base, -2.0f * static_cast<float>(pair) /
                           static_cast<float>(rotary_dimension));
    if (scaling.kind == 1) {
        frequency /= scaling.factor;
    } else if (scaling.kind == 3) {
        const float low = floorf(yarn_correction_dimension(
            scaling.beta_fast, rotary_dimension, theta, scaling.original_context));
        const float high = ceilf(yarn_correction_dimension(
            scaling.beta_slow, rotary_dimension, theta, scaling.original_context));
        const float clipped_low = fmaxf(0.0f, low);
        const float clipped_high = fminf(
            static_cast<float>(rotary_dimension - 1), high);
        const float span = fmaxf(0.001f, clipped_high - clipped_low);
        const float ramp = fminf(1.0f, fmaxf(0.0f,
            (static_cast<float>(pair) - clipped_low) / span));
        const float interpolated = frequency / scaling.factor;
        frequency = frequency * (1.0f - ramp) + interpolated * ramp;
    } else if (scaling.kind == 4) {
        if (pair < scaling.factor_count) {
            const float factor = position > scaling.original_context
                ? scaling.long_factors[pair] : scaling.short_factors[pair];
            frequency /= factor;
        }
    } else if (scaling.kind == 5) {
        const float wavelength = 6.28318530717958647692f / frequency;
        if (wavelength > static_cast<float>(scaling.original_context) /
                         scaling.low_frequency_factor) {
            frequency /= scaling.factor;
        } else if (wavelength > static_cast<float>(scaling.original_context) /
                                      scaling.high_frequency_factor) {
            const float span = scaling.low_frequency_factor -
                scaling.high_frequency_factor;
            const float blend = span > 0.0f
                ? (wavelength * scaling.high_frequency_factor /
                   static_cast<float>(scaling.original_context) - 1.0f) / span
                : 0.0f;
            frequency /= 1.0f + fminf(1.0f, fmaxf(0.0f, blend)) *
                (scaling.factor - 1.0f);
        }
    } else if (scaling.kind == 6) {
        const float fraction = scaling.rotary_fraction > 0.0f
            ? scaling.rotary_fraction : 1.0f;
        frequency = powf(frequency, fraction) / scaling.factor;
    }
    return frequency;
}

}  // namespace celeg::cuda_rope
