#include "detail/compiled_model.hpp"
#include "attention_decode_dispatch.hpp"
#include "attention_kv_store.hpp"
#include "attention_latent_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "attention_qk_prepare.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/moe.hpp"

#include <stdexcept>

namespace celeg {

AttentionCapability CudaCompiledModel::token_attention_plan(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const TokenKvPolicy& kv) {
    return plan_cuda_decode_attention(
        attention.layout,
        resources_.options().kv_cache_mode,
        kv.kv_layout,
        kv.position_source,
        resources_.options().fast_attention,
        kv.paged() && use_segmented_attention(session_.position_),
        attention.alibi_slopes.data() != nullptr,
        owner_layout.head_dim).plan;
}

void CudaCompiledModel::store_and_attend_token_contiguous(
    AttentionLayer& attention, AttentionLayer& owner, const AttentionCapability& plan,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v) {
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    store_standard_attention_kv_contiguous(attention, owner, plan, k, v);

    const auto* block_sparse = std::get_if<BlockSparsePattern>(&layout.pattern);
    dispatch_cuda_contiguous_decode_attention({
        .plan = plan,
        .position_mode = CudaDecodePositionMode::HostScalar,
        .block_sparse = block_sparse,
        .query = q,
        .bf16_kv = cuda_bf16_kv_view(owner),
        .int8_kv = cuda_int8_kv_view(owner),
        .out = workspace_.op_output_.data(),
        .geometry = make_cuda_gqa_geometry(layout, owner_layout),
        .extent = block_sparse
            ? AttentionExtent{.position = position_device_.data()}
            : AttentionExtent{.seq_len = session_.position_ + 1},
        .stream = stream_.get()});
}

void CudaCompiledModel::store_and_attend_token_paged(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const AttentionCapability& plan, int slot, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    const PagedKvIndex index{
        .page_tables = kv.device_page_table,
        .page_table_stride = kv.page_table_stride,
        .attention_slot = slot,
        .page_tokens = paged_kv.page_tokens(),
        .page_vector_elements = paged_kv.page_vector_elements(),
        .layer_vector_offset = paged_kv.layer_vector_offset(slot)};
    const PagedKvScaleIndex scale_index{
        .page_scale_elements = paged_kv.page_scale_elements(),
        .layer_scale_offset = paged_kv.layer_scale_offset(slot)};
    store_standard_attention_kv_paged(owner_layout, plan, slot, k, v, kv);

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
        .index = index,
        .scale_index = scale_index,
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
        .stream = stream_.get()});
}

void CudaCompiledModel::run_token_latent_attention_paged(
    AttentionLayer& attention, const CompiledLayerProgram& semantics,
    int layer_index, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    if (resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 paged state storage");
    }
    if (layout.output_gate.has_value() || layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA latent attention does not support query gates or M-RoPE yet");
    }
    project_cuda_latent_attention_qkv(*this, attention);
    prepare_cuda_latent_attention_qk({
        .layout = &layout,
        .query_rope = workspace_.latent_query_rope_.data(),
        .key_rope = attention.latent_key_rope
            ? workspace_.latent_key_rope_.data() : nullptr,
        .fallback_norm_epsilon = resources_.program_.final_norm.epsilon,
        .position_mode = CudaQkPositionMode::HostScalar,
        .host_position = session_.position_,
        .stream = stream_.get()});
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        attention, layer_index, resources_.layers_);
    const int slot = paged_kv.attention_slot(resolved_owner.model_layer);
    store_cuda_latent_kv_paged(
        *this, attention, paged_kv, slot,
        kv.device_page_table, kv.page_table_stride);
    dispatch_cuda_latent_attention_paged(
        *this, attention, paged_kv, slot,
        kv.device_page_table, kv.page_table_stride);
    project_latent_attention_output(attention, semantics);
}

void CudaCompiledModel::run_token_attention(
    AttentionLayer& attention, const CompiledLayerProgram& semantics,
    int layer_index, const TokenKvPolicy& kv) {
    const AttentionSpec& layout = attention.layout;
    const auto* compiled_attention =
        std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!compiled_attention) {
        throw std::logic_error("CUDA token attention has no compiled attention program");
    }

    if (compiled_attention->execution.kind != AttentionExecutionKind::Standard) {
        if (!kv.paged()) {
            throw std::invalid_argument(
                "CUDA latent attention is not implemented for contiguous host token execution");
        }
        run_token_latent_attention_paged(attention, semantics, layer_index, kv);
        return;
    }

    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        attention, layer_index, resources_.layers_);
    AttentionLayer& owner = *resolved_owner.layer;
    const AttentionSpec& owner_layout = owner.layout;

    const CudaQkvProjectionView qkv = project_standard_attention_qkv(attention);
    __nv_bfloat16* q = qkv.query;
    __nv_bfloat16* k = qkv.key;
    __nv_bfloat16* v = qkv.value;
    const bool output_gate = layout.output_gate.has_value();
    const bool gate_packed = output_gate && layout.output_gate->packed_with_query;

    if (gate_packed) {
        launch_extract_attention_output_gate(
            q, workspace_.q_.data(), workspace_.attention_gate_.data(),
            1, layout.query_width(), layout.head_dim, stream_.get());
        q = workspace_.q_.data();
    }

    const float qk_epsilon = layout.query_norm
        ? layout.query_norm->epsilon
        : (layout.key_norm ? layout.key_norm->epsilon : resources_.program_.final_norm.epsilon);
    const auto* multi = kv.paged() ? nullptr : layout.multi_axis_position();
    if (layout.rope_position() && multi) {
        const auto& position =
            kv.rope_position ? *kv.rope_position : session_.next_rope_position_;
        CELEG_CUDA(cudaMemcpyAsync(mrope_position_device_.data(), position.data(),
                                   sizeof(position), cudaMemcpyHostToDevice,
                                   stream_.get()));
    }

    CudaAttentionQkPreparation qk_preparation{
        .layout = &layout,
        .query = q,
        .key = attention.key ? k : nullptr,
        .query_norm = attention.q_norm,
        .key_norm = attention.k_norm,
        .norm_epsilon = qk_epsilon,
        .position_mode = multi
            ? CudaQkPositionMode::MultiAxisDevice
            : CudaQkPositionMode::HostScalar,
        .host_position = session_.position_,
        .device_position = multi ? mrope_position_device_.data() : nullptr,
        .stream = stream_.get()};
    if (multi) {
        qk_preparation.mrope_section0 = multi->sections[0];
        qk_preparation.mrope_section1 = multi->sections[1];
        qk_preparation.mrope_section2 = multi->sections[2];
        qk_preparation.mrope_interleaved = multi->interleaved;
    }
    prepare_cuda_attention_qk(qk_preparation);

    const AttentionCapability plan = token_attention_plan(attention, owner_layout, kv);
    if (kv.paged()) {
        const int slot = kv.paged_kv->attention_slot(resolved_owner.model_layer);
        if (slot < 0) throw std::logic_error("attention layer has no page slot");
        store_and_attend_token_paged(attention, owner_layout, plan, slot, q, k, v, kv);
    } else {
        store_and_attend_token_contiguous(attention, owner, plan, q, k, v);
    }

    if (output_gate) {
        const __nv_bfloat16* gate = workspace_.attention_gate_.data();
        if (!gate_packed) {
            linear(workspace_.normed_.data(), *attention.gate,
                   workspace_.attention_gate_.data(), 1,
                   layout.query_width(), resources_.program_.hidden);
        }
        launch_sigmoid_multiply(workspace_.op_output_.data(), gate,
                                layout.query_width(), stream_.get());
    }
    project_standard_attention_output(attention, semantics);
}

}
