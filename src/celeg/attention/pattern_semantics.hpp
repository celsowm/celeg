#pragma once

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_PATTERN_SEMANTICS_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_PATTERN_SEMANTICS_INLINE inline
#endif

CELEG_PATTERN_SEMANTICS_INLINE int pattern_max_int(int a, int b) {
    return a > b ? a : b;
}

CELEG_PATTERN_SEMANTICS_INLINE bool causal_visible(
    int query_position, int key_position) {
    return query_position >= 0 && key_position >= 0 &&
           key_position <= query_position;
}

CELEG_PATTERN_SEMANTICS_INLINE int sliding_window_first_candidate(
    int query_position, int window) {
    return pattern_max_int(0, query_position - window + 1);
}

CELEG_PATTERN_SEMANTICS_INLINE bool sliding_window_visible(
    int query_position, int key_position, int window) {
    return causal_visible(query_position, key_position) &&
           key_position >= sliding_window_first_candidate(query_position, window);
}

CELEG_PATTERN_SEMANTICS_INLINE bool prefix_lm_visible(
    int query_position, int key_position, int prefix_length) {
    if (query_position < 0 || key_position < 0) return false;
    return query_position < prefix_length
        ? key_position < prefix_length
        : key_position <= query_position;
}

CELEG_PATTERN_SEMANTICS_INLINE bool block_sparse_visible(
    int query_position, int key_position, int block_size,
    int local_blocks, int global_blocks) {
    if (!causal_visible(query_position, key_position)) return false;
    const int query_block = query_position / block_size;
    const int key_block = key_position / block_size;
    if (key_block < global_blocks) return key_block <= query_block;
    return key_block >= query_block - local_blocks + 1 &&
           key_block <= query_block;
}

CELEG_PATTERN_SEMANTICS_INLINE bool dynamic_sparse_visible(
    int query_position, int key_position, int block_size,
    int max_selected_blocks) {
    if (!causal_visible(query_position, key_position)) return false;
    const int query_block = query_position / block_size;
    const int key_block = key_position / block_size;
    if (key_block == query_block) return true;
    return key_block < max_selected_blocks;
}

CELEG_PATTERN_SEMANTICS_INLINE bool prefix_lm_may_read_future(
    int query_position, int sequence_length, int prefix_length) {
    return query_position < prefix_length && prefix_length < sequence_length;
}

#undef CELEG_PATTERN_SEMANTICS_INLINE

}
