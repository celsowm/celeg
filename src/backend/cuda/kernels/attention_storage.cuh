#pragma once

#include "paged_kv_offsets.cuh"

__device__ __forceinline__ float attention_dot_int8(
    const __nv_bfloat16* query, const int8_t* key, float key_scale,
    int head_dim, float* warp_sums, float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) *
            (static_cast<float>(key[d]) * key_scale);
    }
    return block_sum(partial, warp_sums, total);
}

__device__ __forceinline__ float attention_dot(
    const __nv_bfloat16* query, const __nv_bfloat16* key, int head_dim,
    float* warp_sums, float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) * bf16_float(key[d]);
    }
    return block_sum(partial, warp_sums, total);
}

struct ContiguousBf16AttentionStorage {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int kv_heads;

    __device__ __forceinline__ ContiguousBf16AttentionStorage row(int) const {
        return *this;
    }

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t base =
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return query_value * bf16_float(keys[base + dimension]);
    }

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return attention_dot(query, key, head_dim, warp_sums, dot_total);
    }

    __device__ __forceinline__ float value(
        int token, int kv_head, int dimension, int head_dim) const {
        const __nv_bfloat16* value_ptr = values +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return bf16_float(value_ptr[dimension]);
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        return weight * value(token, kv_head, dimension, head_dim);
    }

    __device__ __forceinline__ float block_sparse_weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        return probability * value(token, kv_head, dimension, head_dim);
    }
};

struct ContiguousInt8AttentionStorage {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    int kv_heads;

    __device__ __forceinline__ ContiguousInt8AttentionStorage row(int) const {
        return *this;
    }

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        return query_value * static_cast<float>(
            keys[scale_index * head_dim + dimension]) * key_scales[scale_index];
    }

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = keys + scale_index * head_dim;
        return attention_dot_int8(
            query, key, key_scales[scale_index], head_dim, warp_sums, dot_total);
    }

    __device__ __forceinline__ float value(
        int token, int kv_head, int dimension, int head_dim) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* value_ptr = values + scale_index * head_dim;
        return static_cast<float>(value_ptr[dimension]) * value_scales[scale_index];
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        return static_cast<float>(values[
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim + dimension]) *
            value_scales[static_cast<size_t>(token) * kv_heads + kv_head] * weight;
    }

    __device__ __forceinline__ float block_sparse_weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        return probability * value(token, kv_head, dimension, head_dim);
    }
};

struct PointerBf16AttentionStorage {
    const __nv_bfloat16* const* keys;
    const __nv_bfloat16* const* values;
    int kv_heads;

    __device__ __forceinline__ ContiguousBf16AttentionStorage row(int index) const {
        return {keys[index], values[index], kv_heads};
    }
};

struct PointerInt8AttentionStorage {
    const int8_t* const* keys;
    const int8_t* const* values;
    const float* const* key_scales;
    const float* const* value_scales;
    int kv_heads;

    __device__ __forceinline__ ContiguousInt8AttentionStorage row(int index) const {
        return {keys[index], values[index], key_scales[index],
                value_scales[index], kv_heads};
    }
};

struct PagedBf16AttentionRow {
    const __nv_bfloat16* key_pool;
    const __nv_bfloat16* value_pool;
    const uint32_t* page_table;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    int kv_heads;

    __device__ __forceinline__ uint32_t page(int token) const {
        return page_table[token / page_tokens];
    }

    __device__ __forceinline__ size_t offset(
        int token, int kv_head, int dimension, int head_dim) const {
        return paged_vector_offset(
            page(token), attention_slot, token % page_tokens, kv_head, dimension,
            page_tokens, page_vector_elements, layer_vector_offset,
            kv_heads, head_dim);
    }

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        return query_value * bf16_float(
            key_pool[offset(token, kv_head, dimension, head_dim)]);
    }

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        float local = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            local += dot_term(bf16_float(query[d]), token, kv_head, d, head_dim);
        }
        return block_sum(local, warp_sums, dot_total);
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        return bf16_float(
            value_pool[offset(token, kv_head, dimension, head_dim)]) * weight;
    }

    __device__ __forceinline__ float block_sparse_weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        return probability * bf16_float(
            value_pool[offset(token, kv_head, dimension, head_dim)]);
    }
};

struct PagedBf16AttentionStorage {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    const uint32_t* page_tables;
    int page_table_stride;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    int kv_heads;

    __device__ __forceinline__ PagedBf16AttentionRow row(int index) const {
        return {keys, values,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, kv_heads};
    }
};

struct PagedInt8AttentionRow {
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

    __device__ __forceinline__ uint32_t page(int token) const {
        return page_table[token / page_tokens];
    }

    __device__ __forceinline__ size_t scale_offset(
        int token, int kv_head) const {
        return paged_scale_offset(
            page(token), attention_slot, token % page_tokens, kv_head,
            page_tokens, page_scale_elements, layer_scale_offset, kv_heads);
    }

    __device__ __forceinline__ size_t offset(
        int token, int kv_head, int dimension, int head_dim) const {
        return paged_vector_offset(
            page(token), attention_slot, token % page_tokens, kv_head, dimension,
            page_tokens, page_vector_elements, layer_vector_offset,
            kv_heads, head_dim);
    }

    __device__ __forceinline__ float dot_term(
        float query_value, int token, int kv_head, int dimension,
        int head_dim) const {
        return query_value * static_cast<float>(
            key_pool[offset(token, kv_head, dimension, head_dim)]) *
            key_scales[scale_offset(token, kv_head)];
    }

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        float local = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            local += dot_term(bf16_float(query[d]), token, kv_head, d, head_dim);
        }
        return block_sum(local, warp_sums, dot_total);
    }

    __device__ __forceinline__ float weighted_value(
        float weight, int token, int kv_head, int dimension,
        int head_dim) const {
        return static_cast<float>(
            value_pool[offset(token, kv_head, dimension, head_dim)]) *
            value_scales[scale_offset(token, kv_head)] * weight;
    }

    __device__ __forceinline__ float block_sparse_weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        return probability * static_cast<float>(
            value_pool[offset(token, kv_head, dimension, head_dim)]) *
            value_scales[scale_offset(token, kv_head)];
    }
};

struct PagedInt8AttentionStorage {
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

    __device__ __forceinline__ PagedInt8AttentionRow row(int index) const {
        return {keys, values, key_scales, value_scales,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, page_scale_elements, layer_scale_offset,
                kv_heads};
    }
};
