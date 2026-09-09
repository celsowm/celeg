#pragma once

#include <cmath>

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_MERGE_SEMANTICS_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_MERGE_SEMANTICS_INLINE inline
#endif

struct MergeTransition {
    float maximum;
    float denominator;
    float destination_scale;
    float source_scale;
};

CELEG_MERGE_SEMANTICS_INLINE float merge_max(float a, float b) {
    return a > b ? a : b;
}

CELEG_MERGE_SEMANTICS_INLINE bool merge_finite(float value) {
#if defined(__CUDA_ARCH__)
    return isfinite(value);
#else
    return std::isfinite(value);
#endif
}

CELEG_MERGE_SEMANTICS_INLINE float merge_exp(float value) {
#if defined(__CUDA_ARCH__)
    return expf(value);
#else
    return std::exp(value);
#endif
}

CELEG_MERGE_SEMANTICS_INLINE float partial_rescale(
    float partial_maximum, float global_maximum) {
    return merge_exp(partial_maximum - global_maximum);
}

CELEG_MERGE_SEMANTICS_INLINE MergeTransition merge_pair(
    float destination_maximum, float destination_denominator,
    float source_maximum, float source_denominator) {
    const float maximum = merge_max(destination_maximum, source_maximum);
    const float destination_scale = merge_finite(destination_maximum)
        ? partial_rescale(destination_maximum, maximum) : 0.0f;
    const float source_scale = partial_rescale(source_maximum, maximum);
    return {
        maximum,
        destination_denominator * destination_scale +
            source_denominator * source_scale,
        destination_scale,
        source_scale};
}

CELEG_MERGE_SEMANTICS_INLINE float merge_accumulate(
    float destination, float source, const MergeTransition& transition) {
    return destination * transition.destination_scale +
        source * transition.source_scale;
}

#undef CELEG_MERGE_SEMANTICS_INLINE

}
