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

__global__ void gqa_prefill_block_sparse_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, __nv_bfloat16* out, int rows,
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
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
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
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_row; ++token) {
        if (!block_sparse_visible(query_row, token, pattern)) continue;
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = values +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator += probability * bf16_float(value[lane]);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_prefill_block_sparse_int8_kernel(
    const __nv_bfloat16* query, const int8_t* keys, const int8_t* values,
    const float* key_scales, const float* value_scales, __nv_bfloat16* out,
    int rows, int q_heads, int kv_heads, int head_dim,
    GqaBlockSparsePattern pattern) {
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
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = keys + scale_index * head_dim;
        const float dot = attention_dot_int8(
            q, key, key_scales[scale_index], head_dim, warp_sums, &dot_total);
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
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = keys + scale_index * head_dim;
        const float dot = attention_dot_int8(
            q, key, key_scales[scale_index], head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_row; ++token) {
        if (!block_sparse_visible(query_row, token, pattern)) continue;
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = keys + scale_index * head_dim;
        const float dot = attention_dot_int8(
            q, key, key_scales[scale_index], head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = values + scale_index * head_dim;
            accumulator += probability *
                (static_cast<float>(value[lane]) * value_scales[scale_index]);
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
    gqa_prefill_block_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, args.extent.rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_block_sparse_int8(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_prefill_block_sparse_int8_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, args.extent.rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
