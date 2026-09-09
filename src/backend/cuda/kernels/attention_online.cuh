#pragma once

/**
 * @brief Runs one warp-online attention row with storage and score policies.
 *
 * The recurrence and arithmetic grouping match the specialized ALiBi and
 * relative-bias kernels this replaces. Callers continue to launch 32 threads.
 */
template <typename RowStorage, typename ScorePolicy>
__device__ __forceinline__ void online_attention_row(
    const __nv_bfloat16* query, RowStorage storage, __nv_bfloat16* output,
    int head, int kv_head, int head_dim, int query_position, int first_token,
    int sequence_length, float scale, ScorePolicy score_policy) {
    const int lane = threadIdx.x;
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;

    for (int token = first_token; token < sequence_length; ++token) {
        float partial = 0.0f;
        for (int dimension = lane; dimension < head_dim; dimension += 32) {
            partial += storage.dot_term(
                bf16_float(query[dimension]), token, kv_head, dimension,
                head_dim);
        }
        const float score = score_policy.score(
            warp_broadcast_sum(partial), scale, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        int index = 0;
        for (int dimension = lane; dimension < head_dim;
             dimension += 32, ++index) {
            accumulator[index] = accumulator[index] * alpha +
                storage.weighted_value(
                    beta, token, kv_head, dimension, head_dim);
        }
        running_max = next_max;
    }

    int index = 0;
    for (int dimension = lane; dimension < head_dim;
         dimension += 32, ++index) {
        output[dimension] = __float2bfloat16(accumulator[index] / denominator);
    }
}

/** @brief Generic 32-thread online-attention kernel. */
template <typename Storage, typename Positions, typename ScorePolicy>
__global__ void gqa_online_attention_kernel(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* output,
    Positions positions, ScorePolicy score_policy, int rows, int q_heads,
    int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = positions.value(row);
    const int sequence_length = query_position + 1;
    const int first_token = sliding_window > 0
        ? max(0, sequence_length - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads);
    const __nv_bfloat16* query_row = query +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    __nv_bfloat16* output_row = output +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    online_attention_row(
        query_row, storage.row(row), output_row, head, kv_head, head_dim,
        query_position, first_token, sequence_length,
        rsqrtf(static_cast<float>(head_dim)), score_policy);
}
