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
        const auto transition = celeg::attention_semantics::online_transition(
            running_max, denominator, score);
        denominator = transition.denominator;
        int index = 0;
        for (int dimension = lane; dimension < head_dim;
             dimension += 32, ++index) {
            accumulator[index] =
                accumulator[index] * transition.previous_scale +
                storage.weighted_value(
                    transition.current_scale, token, kv_head, dimension,
                    head_dim);
        }
        running_max = transition.maximum;
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
    const int sequence_length =
        celeg::attention_semantics::sequence_length_from_query_position(
            query_position);
    const int first_token = sliding_window > 0
        ? celeg::attention_semantics::sliding_window_first_candidate(
              query_position, sliding_window)
        : 0;
    const int kv_head = celeg::attention_semantics::gqa_kv_head(
        head, q_heads, kv_heads);
    const __nv_bfloat16* query_row = query +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    __nv_bfloat16* output_row = output +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    online_attention_row(
        query_row, storage.row(row), output_row, head, kv_head, head_dim,
        query_position, first_token, sequence_length,
        celeg::attention_semantics::attention_scale(head_dim), score_policy);
}
