#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

void launch_gqa_prefill_block_sparse(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, args.geometry.kv_heads};
    gqa_block_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, AttentionPrefillPosition{},
        args.extent.rows, args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_block_sparse_int8(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const int threads = attention_threads(args.geometry.head_dim);
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.geometry.kv_heads};
    gqa_block_sparse_kernel<<<
        args.extent.rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, storage, args.out, AttentionPrefillPosition{},
        args.extent.rows, args.geometry.q_heads, args.geometry.kv_heads,
        args.geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

}
