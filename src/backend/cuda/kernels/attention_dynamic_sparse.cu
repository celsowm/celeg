#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"

namespace {

constexpr int kMaxDynamicSparseBlocks = 32;

__device__ __forceinline__ bool selected_block(
    int block, const int* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        if (blocks[i] == block) return true;
    }
    return false;
}

__device__ __forceinline__ void insert_top_block(
    int candidate_block, float candidate_score,
    int* blocks, float* scores, int count) {
    int slot = -1;
    float minimum = FLT_MAX;
    for (int i = 0; i < count; ++i) {
        if (blocks[i] < 0) {
            slot = i;
            break;
        }
        if (scores[i] < minimum) {
            minimum = scores[i];
            slot = i;
        }
    }
    if (slot >= 0 && (blocks[slot] < 0 || candidate_score > scores[slot])) {
        blocks[slot] = candidate_block;
        scores[slot] = candidate_score;
    }
}

__global__ void gqa_prefill_dynamic_sparse_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* keys,
    const __nv_bfloat16* values, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, GqaDynamicSparsePattern pattern) {
    const int flat = blockIdx.x;
    const int query_row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (query_row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* q = query +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ int selected[kMaxDynamicSparseBlocks];
    __shared__ float selected_scores[kMaxDynamicSparseBlocks];
    __shared__ float block_score;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane < pattern.max_selected_blocks) {
        selected[lane] = -1;
        selected_scores[lane] = -FLT_MAX;
    }
    __syncthreads();

    const int query_block = query_row / pattern.block_size;
    for (int candidate = 0; candidate <= query_block; ++candidate) {
        if (lane == 0) block_score = -FLT_MAX;
        __syncthreads();
        const int begin = candidate * pattern.block_size;
        const int end = min(query_row + 1, begin + pattern.block_size);
        for (int token = begin; token < end; ++token) {
            const __nv_bfloat16* key = keys +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
            if (lane == 0) block_score = fmaxf(block_score, dot * scale);
            __syncthreads();
        }
        if (lane == 0) {
            insert_top_block(candidate, block_score, selected, selected_scores,
                             pattern.max_selected_blocks);
        }
        __syncthreads();
    }

    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token <= query_row; ++token) {
        if (!selected_block(token / pattern.block_size, selected,
                            pattern.max_selected_blocks)) continue;
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) maximum = fmaxf(
            maximum, rounded_bf16_float(rounded_bf16_float(dot) * scale));
        __syncthreads();
    }

    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token <= query_row; ++token) {
        if (!selected_block(token / pattern.block_size, selected,
                            pattern.max_selected_blocks)) continue;
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) denominator += expf(
            rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum);
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_row; ++token) {
        if (!selected_block(token / pattern.block_size, selected,
                            pattern.max_selected_blocks)) continue;
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(q, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) probability = rounded_bf16_float(expf(
            rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum) /
            denominator);
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

}

void launch_gqa_prefill_dynamic_sparse(
    const GqaContiguousArgs& args, GqaDynamicSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_prefill_dynamic_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, args.extent.rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
