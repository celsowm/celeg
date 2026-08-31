#pragma once

__device__ __forceinline__ size_t paged_vector_offset(
    uint32_t page, int attention_slot, int in_page, int head, int dim,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    int kv_heads, int head_dim) {
    return static_cast<size_t>(page) * page_vector_elements + layer_vector_offset +
        ((static_cast<size_t>(in_page) * kv_heads + head) * head_dim) + dim;
}

__device__ __forceinline__ size_t paged_scale_offset(
    uint32_t page, int attention_slot, int in_page, int head,
    int page_tokens, size_t page_scale_elements, size_t layer_scale_offset,
    int kv_heads) {
    return static_cast<size_t>(page) * page_scale_elements + layer_scale_offset +
        static_cast<size_t>(in_page) * kv_heads + head;
}
