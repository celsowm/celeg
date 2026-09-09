#pragma once

#include "attention_storage.cuh"
#include "celeg/attention/pattern_semantics.hpp"

struct AttentionPrefillPosition {
    __device__ __forceinline__ int value(int row) const { return row; }
};

struct AttentionSinglePosition {
    const int32_t* position;
    __device__ __forceinline__ int value(int) const { return *position; }
};

struct AttentionBatchPositions {
    const int32_t* positions;
    __device__ __forceinline__ int value(int row) const { return positions[row]; }
};

__device__ __forceinline__ bool attention_block_sparse_visible(
    int query_position, int token, const GqaBlockSparsePattern& pattern) {
    return celeg::attention_semantics::block_sparse_visible(
        query_position, token, pattern.block_size,
        pattern.local_blocks, pattern.global_blocks);
}

/// Online/segmented decode kernels launch with 32 threads and stride
/// `for (d = lane; d < head_dim; d += 32)`, accumulating into a per-lane
/// register array. The supported head_dim ceiling is therefore
/// `kMaxHeadDimPerLane * 32`; 16 covers head_dim 512 (e.g. Gemma full-attention
/// layers) and anything larger must fail in `validate_cuda_attention_semantics`.
constexpr int kMaxHeadDimPerLane = 16;

__device__ __forceinline__ float warp_broadcast_sum(float partial) {
    return __shfl_sync(0xffffffffu, warp_sum(partial), 0);
}

__device__ __forceinline__ float merge_segmented_attention_lane(
    size_t base, int count, int lane, int head_dim,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    float global_max = -FLT_MAX;
    for (int segment = 0; segment < count; ++segment) {
        global_max = fmaxf(global_max, partial_max[base + segment]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int segment = 0; segment < count; ++segment) {
        const float local_denom = partial_denom[base + segment];
        if (local_denom == 0.0f) continue;
        const float factor = celeg::attention_semantics::partial_rescale(
            partial_max[base + segment], global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            const size_t accum_index =
                (base + segment) * static_cast<size_t>(head_dim) + lane;
            accumulator += partial_accum[accum_index] * factor;
        }
    }
    return accumulator / denominator;
}
