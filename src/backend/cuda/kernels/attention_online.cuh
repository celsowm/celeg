#pragma once

#include "paged_kv_offsets.cuh"

/** @brief Contiguous BF16 KV row used by warp-online attention. */
struct OnlineContiguousBf16Row {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int kv_heads;

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return query_value * bf16_float(keys[base + dimension]);
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return bf16_float(values[base + dimension]) * weight;
    }
};

/** @brief Contiguous INT8 KV row preserving scale multiplication order. */
struct OnlineContiguousInt8Row {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    int kv_heads;

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        return query_value * static_cast<float>(
            keys[scale_index * head_dim + dimension]) * key_scales[scale_index];
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        return static_cast<float>(values[scale_index * head_dim + dimension]) *
            value_scales[scale_index] * weight;
    }
};

/** @brief Paged BF16 KV row used by warp-online attention. */
struct OnlinePagedBf16Row {
    const __nv_bfloat16* key_pool;
    const __nv_bfloat16* value_pool;
    const uint32_t* page_table;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    int kv_heads;

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const uint32_t page = page_table[token / page_tokens];
        const int in_page = token % page_tokens;
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, dimension, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
        return query_value * bf16_float(key_pool[offset]);
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        const uint32_t page = page_table[token / page_tokens];
        const int in_page = token % page_tokens;
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, dimension, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
        return bf16_float(value_pool[offset]) * weight;
    }
};

/** @brief Paged INT8 KV row preserving scale multiplication order. */
struct OnlinePagedInt8Row {
    const int8_t* key_pool;
    const int8_t* value_pool;
    const float* key_scales;
    const float* value_scales;
    const uint32_t* page_table;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    size_t page_scale_elements;
    size_t layer_scale_offset;
    int kv_heads;

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const uint32_t page = page_table[token / page_tokens];
        const int in_page = token % page_tokens;
        const size_t scale_offset = paged_scale_offset(
            page, attention_slot, in_page, kv_head, page_tokens,
            page_scale_elements, layer_scale_offset, kv_heads);
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, dimension, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
        return query_value * static_cast<float>(key_pool[offset]) *
            key_scales[scale_offset];
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        const uint32_t page = page_table[token / page_tokens];
        const int in_page = token % page_tokens;
        const size_t scale_offset = paged_scale_offset(
            page, attention_slot, in_page, kv_head, page_tokens,
            page_scale_elements, layer_scale_offset, kv_heads);
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, dimension, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
        return static_cast<float>(value_pool[offset]) *
            value_scales[scale_offset] * weight;
    }
};

struct OnlineContiguousBf16Storage {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int kv_heads;

    __device__ __forceinline__ OnlineContiguousBf16Row row(int) const {
        return {keys, values, kv_heads};
    }
};

struct OnlineContiguousInt8Storage {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    int kv_heads;

    __device__ __forceinline__ OnlineContiguousInt8Row row(int) const {
        return {keys, values, key_scales, value_scales, kv_heads};
    }
};

struct OnlinePtrBf16Storage {
    const __nv_bfloat16* const* keys;
    const __nv_bfloat16* const* values;
    int kv_heads;

    __device__ __forceinline__ OnlineContiguousBf16Row row(int index) const {
        return {keys[index], values[index], kv_heads};
    }
};

struct OnlinePtrInt8Storage {
    const int8_t* const* keys;
    const int8_t* const* values;
    const float* const* key_scales;
    const float* const* value_scales;
    int kv_heads;

    __device__ __forceinline__ OnlineContiguousInt8Row row(int index) const {
        return {keys[index], values[index], key_scales[index],
                value_scales[index], kv_heads};
    }
};

struct OnlinePagedBf16Storage {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    const uint32_t* page_tables;
    int page_table_stride;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    int kv_heads;

    __device__ __forceinline__ OnlinePagedBf16Row row(int index) const {
        return {keys, values,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, kv_heads};
    }
};

struct OnlinePagedInt8Storage {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    const uint32_t* page_tables;
    int page_table_stride;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    size_t page_scale_elements;
    size_t layer_scale_offset;
    int kv_heads;

    __device__ __forceinline__ OnlinePagedInt8Row row(int index) const {
        return {keys, values, key_scales, value_scales,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, page_scale_elements, layer_scale_offset,
                kv_heads};
    }
};

struct OnlinePrefillPosition {
    __device__ __forceinline__ int value(int row) const { return row; }
};

struct OnlineSinglePosition {
    const int32_t* position;
    __device__ __forceinline__ int value(int) const { return *position; }
};

struct OnlineBatchPositions {
    const int32_t* positions;
    __device__ __forceinline__ int value(int row) const { return positions[row]; }
};

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
