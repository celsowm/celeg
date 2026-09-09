#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

namespace {

template <typename Storage>
__global__ void gqa_decode_block_sparse_kernel(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    GqaBlockSparsePattern pattern) {
    const int query_head = blockIdx.x;
    if (query_head >= q_heads) return;

    const int query_position = *position;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* q = query +
        static_cast<size_t>(query_head) * head_dim;
    __nv_bfloat16* output = out +
        static_cast<size_t>(query_head) * head_dim;

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    block_sparse_attention_row(
        q, storage, output, query_position, kv_head, head_dim, pattern,
        warp_sums, &dot_total, &maximum, &denominator, &probability);
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
