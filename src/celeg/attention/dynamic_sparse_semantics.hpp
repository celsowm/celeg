#pragma once

namespace celeg::attention_semantics {

#if defined(__CUDACC__)
#define CELEG_DYNAMIC_SPARSE_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_DYNAMIC_SPARSE_INLINE inline
#endif

// DynamicSparsePattern is content-ranked, not a position-only visibility mask.
// Every causal block is ranked by the maximum scaled Q.K score among its
// causally visible tokens. At most max_selected_blocks are retained; equal
// scores prefer the lower block index so CPU reference code and accelerators
// can make exactly the same selection.
CELEG_DYNAMIC_SPARSE_INLINE bool dynamic_sparse_score_better(
    float candidate_score, int candidate_block,
    float incumbent_score, int incumbent_block) {
    if (incumbent_block < 0) return true;
    if (candidate_score > incumbent_score) return true;
    if (candidate_score < incumbent_score) return false;
    return candidate_block < incumbent_block;
}

CELEG_DYNAMIC_SPARSE_INLINE float dynamic_sparse_lowest_score() {
    return -3.402823466e+38F;
}

CELEG_DYNAMIC_SPARSE_INLINE void dynamic_sparse_initialize(
    int* blocks, float* scores, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i] = -1;
        scores[i] = dynamic_sparse_lowest_score();
    }
}

// Keeps the selected slots ordered best-to-worst. This is not required by the
// attention kernel, but makes tie behavior and cross-backend conformance
// deterministic and directly testable.
CELEG_DYNAMIC_SPARSE_INLINE void dynamic_sparse_insert_top_block(
    int candidate_block, float candidate_score,
    int* blocks, float* scores, int count) {
    if (count <= 0 || candidate_block < 0) return;
    int insert_at = count;
    for (int i = 0; i < count; ++i) {
        if (dynamic_sparse_score_better(
                candidate_score, candidate_block, scores[i], blocks[i])) {
            insert_at = i;
            break;
        }
    }
    if (insert_at == count) return;
    for (int i = count - 1; i > insert_at; --i) {
        blocks[i] = blocks[i - 1];
        scores[i] = scores[i - 1];
    }
    blocks[insert_at] = candidate_block;
    scores[insert_at] = candidate_score;
}

CELEG_DYNAMIC_SPARSE_INLINE bool dynamic_sparse_selected_block(
    int block, const int* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        if (blocks[i] == block) return true;
    }
    return false;
}

CELEG_DYNAMIC_SPARSE_INLINE int dynamic_sparse_query_block(
    int query_position, int block_size) {
    return query_position / block_size;
}

CELEG_DYNAMIC_SPARSE_INLINE int dynamic_sparse_candidate_count(
    int query_position, int block_size) {
    return dynamic_sparse_query_block(query_position, block_size) + 1;
}

CELEG_DYNAMIC_SPARSE_INLINE int dynamic_sparse_block_begin(
    int block, int block_size) {
    return block * block_size;
}

CELEG_DYNAMIC_SPARSE_INLINE int dynamic_sparse_block_end_exclusive(
    int query_position, int block, int block_size) {
    const int block_end = (block + 1) * block_size;
    const int causal_end = query_position + 1;
    return block_end < causal_end ? block_end : causal_end;
}

CELEG_DYNAMIC_SPARSE_INLINE void dynamic_sparse_select_top_k(
    const float* candidate_scores, int candidate_count,
    int max_selected_blocks, int* blocks, float* scores) {
    dynamic_sparse_initialize(blocks, scores, max_selected_blocks);
    for (int block = 0; block < candidate_count; ++block) {
        dynamic_sparse_insert_top_block(
            block, candidate_scores[block], blocks, scores,
            max_selected_blocks);
    }
}

#undef CELEG_DYNAMIC_SPARSE_INLINE

}
