#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"

namespace {

__device__ __forceinline__ bool block_sparse_decode_visible(
    int query_position, int token, const GqaBlockSparsePattern& pattern) {
    if (token > query_position) return false;
    const int query_block = query_position / pattern.block_size;
    const int token_block = token / pattern.block_size;
    if (token_block < pattern.global_blocks) return true;
    const int local_start = max(0, query_block - pattern.local_blocks + 1);
    return token_block >= local_start && token_block <= query_block;
}

template <typename Storage>
__global__ void gqa_decode_block_sparse_kernel(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    GqaBlockSparsePattern pattern) {
    const int query_head = blockIdx.x;
    if (query_head >= q_heads) return;

    const int query_position = *position;
    const int seq_len = query_position + 1;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* q = query +
        static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token < seq_len; ++token) {
        if (!block_sparse_decode_visible(query_position, token, pattern)) continue;
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
    for (int token = 0; token < seq_len; ++token) {
        if (!block_sparse_decode_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            q, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        if (!block_sparse_decode_visible(query_position, token, pattern)) continue;
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
        out[static_cast<size_t>(query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

}

void launch_gqa_decode_block_sparse_device(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, args.geometry.kv_heads};
    gqa_decode_block_sparse_kernel<<<
        args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_block_sparse_int8_device(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.geometry.kv_heads};
    gqa_decode_block_sparse_kernel<<<
        args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
