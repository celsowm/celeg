#include "kernel_common.cuh"
#include "kernels/attention.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

void launch_gqa_decode_block_sparse_device(
    const GqaContiguousArgs& args, GqaBlockSparsePattern pattern) {
    const ContiguousBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out,
        AttentionSinglePosition{args.extent.position}, 1,
        args.geometry, pattern, args.stream);
}

void launch_gqa_decode_block_sparse_int8_device(
    const GqaContiguousInt8Args& args, GqaBlockSparsePattern pattern) {
    const ContiguousInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out,
        AttentionSinglePosition{args.extent.position}, 1,
        args.geometry, pattern, args.stream);
}

}
