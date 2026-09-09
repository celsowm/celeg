#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"

namespace {

__device__ __forceinline__ bool block_sparse_visible(
    int query_row, int token, const GqaBlockSparsePattern& pattern) {
    if (token > query_row) return false;
    const int query_block = query_row / pattern.block_size;
    const int token_block = token / pattern.block_size;
    if (token_block < pattern.global_blocks) return true;
    const int local_start = max(0, query_block - pattern.local_blocks + 1);
    return token_block >= local_start && token_block <= query_block;
}

struct BlockSparseBf16Storage {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int kv_heads;

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
};

struct BlockSparseInt8Storage {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    int kv_heads;

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
};

template <typename Storage>
__global__ void gqa_prefill_block_sparse_kernel(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, GqaBlockSparsePattern pattern) {
    const int block = blockIdx.x;
    const int query_row = block / q_heads;
    const int query_head = block % q_heads;
    if (query_row >= rows || query_head >= q_heads) return;

    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* q = query +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token <= query_row; ++token) {
        if (!block_sparse_visible(query_row, token, pattern)) continue;
        const float dot = storage.dot(
            q, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }

    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token <= query_row; ++token) {
        if (!block_sparse_visible(query_row, token, pattern)) continue;
        const float dot = storage.dot(
            q, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_row; ++token) {
        if (!block_sparse_visible(query_row, token, pattern)) continue;
        const float dot = storage.dot(
            q, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            accumulator += probability *
                storage.value(token, kv_head, lane, head_dim);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

}

void launch_gqa_prefill_block_sparse(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const BlockSparseBf16Storage storage{
        args.kv.keys, args.kv.values, args.geometry.kv_heads};
    gqa_prefill_block_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, args.extent.rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_block_sparse_int8(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const BlockSparseInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.geometry.kv_heads};
    gqa_prefill_block_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, args.extent.rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
