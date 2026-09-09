#pragma once

#include <cmath>

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_ATTENTION_SEMANTICS_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_ATTENTION_SEMANTICS_INLINE inline
#endif

CELEG_ATTENTION_SEMANTICS_INLINE int max_int(int a, int b) {
    return a > b ? a : b;
}

CELEG_ATTENTION_SEMANTICS_INLINE int min_int(int a, int b) {
    return a < b ? a : b;
}

CELEG_ATTENTION_SEMANTICS_INLINE int abs_int(int value) {
    return value < 0 ? -value : value;
}

CELEG_ATTENTION_SEMANTICS_INLINE float natural_log(float value) {
#if defined(__CUDA_ARCH__)
    return logf(value);
#else
    return std::log(value);
#endif
}

CELEG_ATTENTION_SEMANTICS_INLINE int relative_position_bucket(
    int query_position, int key_position, int total_bucket_count,
    int max_distance, bool bidirectional) {
    const int relative_position = key_position - query_position;
    const int bucket_count = bidirectional
        ? total_bucket_count / 2 : total_bucket_count;
    const bool positive = bidirectional && relative_position > 0;
    const int distance = bidirectional
        ? abs_int(relative_position) : max_int(-relative_position, 0);
    const int max_exact = bucket_count / 2;
    int bucket = 0;
    if (distance < max_exact) {
        bucket = distance;
    } else {
        const int safe_exact = max_int(max_exact, 1);
        const int safe_distance = max_int(distance, max_exact);
        const int safe_max_distance = max_int(max_distance, max_exact + 1);
        const float denominator = natural_log(
            static_cast<float>(safe_max_distance) /
            static_cast<float>(safe_exact));
        const float logarithmic = denominator == 0.0f ? 0.0f : natural_log(
            static_cast<float>(safe_distance) /
            static_cast<float>(safe_exact)) / denominator;
        bucket = max_exact + static_cast<int>(
            logarithmic * static_cast<float>(bucket_count - max_exact));
        bucket = min_int(bucket, bucket_count - 1);
    }
    if (positive) bucket += bucket_count;
    return bucket;
}

CELEG_ATTENTION_SEMANTICS_INLINE int alibi_distance(
    int query_position, int key_position) {
    return abs_int(query_position - key_position);
}

CELEG_ATTENTION_SEMANTICS_INLINE float alibi_bias(
    float slope, int query_position, int key_position) {
    return -slope * static_cast<float>(
        alibi_distance(query_position, key_position));
}

#undef CELEG_ATTENTION_SEMANTICS_INLINE

}
