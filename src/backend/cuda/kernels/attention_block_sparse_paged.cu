#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "paged_kv_offsets.cuh"
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

namespace {

struct PagedBlockSparseBf16Row {
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

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const int in_page = token % page_tokens;
        const uint32_t physical_page = page(token);
        float local = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const size_t offset = paged_vector_offset(
                physical_page, attention_slot, in_page, kv_head, d, page_tokens,
                page_vector_elements, layer_vector_offset, kv_heads, head_dim);
            local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
        }
        return block_sum(local, warp_sums, dot_total);
    }

    __device__ __forceinline__ float weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        const int in_page = token % page_tokens;
        const uint32_t physical_page = page(token);
        const size_t offset = paged_vector_offset(
            physical_page, attention_slot, in_page, kv_head, dimension,
            page_tokens, page_vector_elements, layer_vector_offset,
            kv_heads, head_dim);
        return probability * bf16_float(value_pool[offset]);
    }
};

struct PagedBlockSparseBf16Storage {
    const __nv_bfloat16* key_pool;
    const __nv_bfloat16* value_pool;
    const uint32_t* page_tables;
    int page_table_stride;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    int kv_heads;

    __device__ __forceinline__ PagedBlockSparseBf16Row row(int index) const {
        return {key_pool, value_pool,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, kv_heads};
    }
};

struct PagedBlockSparseInt8Row {
    const int8_t* key_pool;
    const int8_t* value_pool;
    const float* key_scale_pool;
    const float* value_scale_pool;
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

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const int in_page = token % page_tokens;
        const uint32_t physical_page = page(token);
        const size_t scale_offset = paged_scale_offset(
            physical_page, attention_slot, in_page, kv_head, page_tokens,
            page_scale_elements, layer_scale_offset, kv_heads);
        float local = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const size_t offset = paged_vector_offset(
                physical_page, attention_slot, in_page, kv_head, d, page_tokens,
                page_vector_elements, layer_vector_offset, kv_heads, head_dim);
            local += bf16_float(query[d]) * static_cast<float>(key_pool[offset]) *
                     key_scale_pool[scale_offset];
        }
        return block_sum(local, warp_sums, dot_total);
    }

    __device__ __forceinline__ float weighted_value(
        float probability, int token, int kv_head, int dimension,
        int head_dim) const {
        const int in_page = token % page_tokens;
        const uint32_t physical_page = page(token);
        const size_t scale_offset = paged_scale_offset(
            physical_page, attention_slot, in_page, kv_head, page_tokens,
            page_scale_elements, layer_scale_offset, kv_heads);
        const size_t offset = paged_vector_offset(
            physical_page, attention_slot, in_page, kv_head, dimension,
            page_tokens, page_vector_elements, layer_vector_offset,
            kv_heads, head_dim);
        return probability * static_cast<float>(value_pool[offset]) *
               value_scale_pool[scale_offset];
    }
};

struct PagedBlockSparseInt8Storage {
    const int8_t* key_pool;
    const int8_t* value_pool;
    const float* key_scale_pool;
    const float* value_scale_pool;
    const uint32_t* page_tables;
    int page_table_stride;
    int attention_slot;
    int page_tokens;
    size_t page_vector_elements;
    size_t layer_vector_offset;
    size_t page_scale_elements;
    size_t layer_scale_offset;
    int kv_heads;

    __device__ __forceinline__ PagedBlockSparseInt8Row row(int index) const {
        return {key_pool, value_pool, key_scale_pool, value_scale_pool,
                page_tables + static_cast<size_t>(index) * page_table_stride,
                attention_slot, page_tokens, page_vector_elements,
                layer_vector_offset, page_scale_elements, layer_scale_offset,
                kv_heads};
    }
};

}

void launch_gqa_decode_block_sparse_paged(
    const GqaPagedArgs& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const PagedBlockSparseBf16Storage storage{
        args.kv.keys, args.kv.values, args.index.page_tables,
        args.index.page_table_stride, args.index.attention_slot,
        args.index.page_tokens, args.index.page_vector_elements,
        args.index.layer_vector_offset, args.geometry.kv_heads};
    gqa_block_sparse_kernel<<<
        args.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        args.rows, args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_block_sparse_int8_paged(
    const GqaPagedInt8Args& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const PagedBlockSparseInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.index.page_tables,
        args.index.page_table_stride, args.index.attention_slot,
        args.index.page_tokens, args.index.page_vector_elements,
        args.index.layer_vector_offset, args.scale_index.page_scale_elements,
        args.scale_index.layer_scale_offset, args.geometry.kv_heads};
    gqa_block_sparse_kernel<<<
        args.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        args.rows, args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
