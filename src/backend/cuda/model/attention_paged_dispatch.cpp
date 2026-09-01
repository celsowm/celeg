#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "backend/cuda/paged_kv.hpp"

namespace celeg {

void CudaCompiledModel::dispatch_standard_attention_paged(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const AttentionCapability& plan, int slot, __nv_bfloat16* q,
    const TokenKvPolicy& kv) {
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    const AttentionSpec& layout = attention.layout;
    const int chunks = (session_.position_ + 1 +
        resources_.options().attention_chunk_tokens - 1) /
        resources_.options().attention_chunk_tokens;

    dispatch_cuda_paged_decode_attention({
        .plan = plan,
        .block_sparse = std::get_if<BlockSparsePattern>(&layout.pattern),
        .query = q,
        .bf16_kv = {.keys = paged_kv.key_bf16(),
                    .values = paged_kv.value_bf16()},
        .int8_kv = {.keys = paged_kv.key_int8(),
                    .values = paged_kv.value_int8(),
                    .key_scales = paged_kv.key_scales(),
                    .value_scales = paged_kv.value_scales()},
        .index = {.page_tables = kv.device_page_table,
                  .page_table_stride = kv.page_table_stride,
                  .attention_slot = slot,
                  .page_tokens = paged_kv.page_tokens(),
                  .page_vector_elements = paged_kv.page_vector_elements(),
                  .layer_vector_offset = paged_kv.layer_vector_offset(slot)},
        .scale_index = {.page_scale_elements = paged_kv.page_scale_elements(),
                        .layer_scale_offset = paged_kv.layer_scale_offset(slot)},
        .out = workspace_.op_output_.data(),
        .positions = position_device_.data(),
        .rows = 1,
        .geometry = make_cuda_gqa_geometry(layout, owner_layout),
        .segmentation = {
            .chunk_tokens = resources_.options().attention_chunk_tokens,
            .chunks = chunks,
            .partial_max = workspace_.attention_partial_max_.data(),
            .partial_denom = workspace_.attention_partial_denom_.data(),
            .partial_accum = workspace_.attention_partial_accum_.data()},
        .alibi_slopes = attention.alibi_slopes.data(),
        .relative_bias = cuda_relative_position_bias_view(attention),
        .stream = stream_.get()});
}

}
