#pragma once

#include <cmath>

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_ONLINE_SEMANTICS_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_ONLINE_SEMANTICS_INLINE inline
#endif

struct OnlineTransition {
    float maximum;
    float denominator;
    float previous_scale;
    float current_scale;
};

CELEG_ONLINE_SEMANTICS_INLINE float online_max(float a, float b) {
    return a > b ? a : b;
}

CELEG_ONLINE_SEMANTICS_INLINE bool online_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

CELEG_ONLINE_SEMANTICS_INLINE float online_exp(float value) {
#if defined(__CUDA_ARCH__)
    return expf(value);
#else
    return std::exp(value);
#endif
}

CELEG_ONLINE_SEMANTICS_INLINE OnlineTransition online_transition(
    float previous_maximum, float previous_denominator, float score) {
    const float maximum = online_max(previous_maximum, score);
    const float previous_scale = online_finite(previous_maximum)
        ? online_exp(previous_maximum - maximum) : 0.0f;
    const float current_scale = online_exp(score - maximum);
    return {
        maximum,
        previous_denominator * previous_scale + current_scale,
        previous_scale,
        current_scale};
}

CELEG_ONLINE_SEMANTICS_INLINE float online_accumulate(
    float accumulator, float value, const OnlineTransition& transition) {
    return accumulator * transition.previous_scale +
        value * transition.current_scale;
}

#undef CELEG_ONLINE_SEMANTICS_INLINE

}
