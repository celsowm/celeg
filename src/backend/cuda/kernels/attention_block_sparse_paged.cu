#include "kernel_common.cuh"
#include "kernels/attention.hpp"
#include "celeg/attention/merge_semantics.hpp"
#include "celeg/attention/pattern_semantics.hpp"

#include <cfloat>

namespace celeg {
#include "attention_common.cuh"
#include "attention_block_sparse_core.cuh"

void launch_gqa_decode_block_sparse_paged(
    const GqaPagedArgs& args, GqaBlockSparsePattern pattern) {
    const PagedBf16AttentionStorage storage{
        args.kv.keys, args.kv.values, args.index.page_tables,
        args.index.page_table_stride, args.index.attention_slot,
        args.index.page_tokens, args.index.page_vector_elements,
        args.index.layer_vector_offset, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        args.rows, args.geometry, pattern, args.stream);
}

void launch_gqa_decode_block_sparse_int8_paged(
    const GqaPagedInt8Args& args, GqaBlockSparsePattern pattern) {
    const PagedInt8AttentionStorage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.index.page_tables,
        args.index.page_table_stride, args.index.attention_slot,
        args.index.page_tokens, args.index.page_vector_elements,
        args.index.layer_vector_offset, args.scale_index.page_scale_elements,
        args.scale_index.layer_scale_offset, args.geometry.kv_heads};
    launch_block_sparse(
        args.query, storage, args.out, AttentionBatchPositions{args.positions},
        args.rows, args.geometry, pattern, args.stream);
}

}
