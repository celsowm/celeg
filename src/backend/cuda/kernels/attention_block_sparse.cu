#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

void launch_gqa_prefill_block_sparse(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern) {
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out, AttentionPrefillPosition{},
        args.extent.rows, args.geometry, pattern, args.stream);
}

void launch_gqa_prefill_block_sparse_int8(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out, AttentionPrefillPosition{},
        args.extent.rows, args.geometry, pattern, args.stream);
}

}
