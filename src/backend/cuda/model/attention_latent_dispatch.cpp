#include "attention_latent_dispatch.hpp"

#include "detail/compiled_model.hpp"
#include "backend/cuda/paged_kv.hpp"
#include "kernels/kernels.cuh"

namespace celeg {
namespace {

LatentQueryView latent_query_view(CudaCompiledModel& model, const AttentionSpec& layout) {
    const auto& latent = *layout.latent_state();
    return {
        .content = model.workspace_.latent_query_content_.data(),
        .rope = latent.factorized() || layout.latent_query_rope_width() != 0
            ? model.workspace_.latent_query_rope_.data() : nullptr};
}

LatentGeometry latent_geometry(const AttentionSpec& layout) {
    const auto& latent = *layout.latent_state();
    return {
        .query_heads = layout.query_heads,
        .latent_rank = latent.latent_rank,
        .rotary_width = latent.factorized()
            ? latent.rope_head_dim
            : (latent.decoupled_rope ? latent.rope_head_dim : 0),
        .score_scale = layout.query_scale,
        .sliding_window = layout.sliding_window_size()};
}

LatentQueryView latent_prefill_query_view(
    CudaCompiledModel& model, const AttentionSpec& layout) {
    const auto& latent = *layout.latent_state();
    return {
        .content = model.workspace_.prefill_latent_query_content_.data(),
        .rope = latent.factorized() || layout.latent_query_rope_width() != 0
            ? model.workspace_.prefill_latent_query_rope_.data() : nullptr};
}

} // namespace

void dispatch_cuda_latent_attention_contiguous(
    CudaCompiledModel& model, AttentionLayer& attention, AttentionLayer& owner) {
    const AttentionSpec& layout = attention.layout;
    launch_latent_attention_device({
        .query = latent_query_view(model, layout),
        .kv = {.keys = owner.latent_key_cache_ptr(),
               .values = owner.latent_value_cache_ptr(),
               .key_rope = owner.latent_key_rope_cache_ptr()},
        .out = model.workspace_.op_output_.data(),
        .extent = {.position = model.position_device_.data()},
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = latent_geometry(layout),
        .stream = model.stream_.get()});
}

void dispatch_cuda_latent_attention_paged(
    CudaCompiledModel& model, AttentionLayer& attention,
    PhysicalPagedKvCache& paged_kv, int slot,
    const uint32_t* device_page_table, int page_table_stride) {
    const AttentionSpec& layout = attention.layout;
    launch_latent_attention_paged_batch({
        .query = latent_query_view(model, layout),
        .kv = {.keys = paged_kv.key_bf16(), .values = paged_kv.value_bf16()},
        .index = {.page_tables = device_page_table,
                  .page_table_stride = page_table_stride,
                  .attention_slot = slot,
                  .page_tokens = paged_kv.page_tokens(),
                  .page_vector_elements = paged_kv.page_vector_elements(),
                  .layer_vector_offset = paged_kv.layer_vector_offset(slot)},
        .out = model.workspace_.op_output_.data(),
        .positions = model.position_device_.data(),
        .rows = 1,
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = latent_geometry(layout),
        .stream = model.stream_.get()});
}

void dispatch_cuda_latent_attention_prefill(
    CudaCompiledModel& model, AttentionLayer& attention,
    AttentionLayer& owner, int rows) {
    const AttentionSpec& layout = attention.layout;
    launch_latent_attention_prefill({
        .query = latent_prefill_query_view(model, layout),
        .kv = {.keys = owner.latent_key_cache_ptr(),
               .values = owner.latent_value_cache_ptr(),
               .key_rope = owner.latent_key_rope_cache_ptr()},
        .out = model.workspace_.prefill_op_output_.data(),
        .extent = {.rows = rows},
        .alibi_slopes = attention.alibi_slopes.data(),
        .geometry = latent_geometry(layout),
        .stream = model.stream_.get()});
}

}
