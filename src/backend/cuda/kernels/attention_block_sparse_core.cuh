#pragma once

template <typename Storage>
__device__ __forceinline__ void block_sparse_attention_row(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* output,
    int query_position, int kv_head, int head_dim,
    const GqaBlockSparsePattern& pattern, float* warp_sums, float* dot_total,
    float* maximum, float* denominator, float* probability) {
    const int lane = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (lane == 0) *maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *maximum = fmaxf(*maximum, score);
        }
        __syncthreads();
    }

    if (lane == 0) *denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *denominator += expf(score - *maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *probability = rounded_bf16_float(
                expf(score - *maximum) / *denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            accumulator += storage.weighted_value(
                *probability, token, kv_head, lane, head_dim);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        output[lane] = __float2bfloat16(accumulator);
    }
}
