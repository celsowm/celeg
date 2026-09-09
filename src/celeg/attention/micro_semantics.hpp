#pragma once

#include <cmath>

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_ATTENTION_MICRO_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_ATTENTION_MICRO_INLINE inline
#endif

CELEG_ATTENTION_MICRO_INLINE int gqa_kv_head(
    int query_head, int query_heads, int kv_heads) {
    return query_head / (query_heads / kv_heads);
}

CELEG_ATTENTION_MICRO_INLINE int sequence_length_from_query_position(
    int query_position) {
    return query_position + 1;
}

CELEG_ATTENTION_MICRO_INLINE int query_position_from_sequence_length(
    int sequence_length) {
    return sequence_length - 1;
}

CELEG_ATTENTION_MICRO_INLINE float attention_scale(int head_dim) {
#if defined(__CUDA_ARCH__)
    return rsqrtf(static_cast<float>(head_dim));
#else
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
#endif
}

#undef CELEG_ATTENTION_MICRO_INLINE

}
