#include "detail/compiled_model.hpp"
#include "attention_contiguous_dispatch.hpp"
#include "attention_decode_dispatch.hpp"
#include "attention_kv_store.hpp"
#include "attention_latent_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "attention_output_gate.hpp"
#include "attention_projection.hpp"
#include "attention_qk_prepare.hpp"
#include "attention_token_qk.hpp"
#include "backend/cuda/paged_kv.hpp"
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
    store_standard_attention_kv_contiguous(attention, owner, plan, k, v);
    dispatch_cuda_standard_attention_contiguous(*this, attention, owner, plan, q);
}

void CudaCompiledModel::store_and_attend_token_paged(
    AttentionLayer& attention, const AttentionSpec& owner_layout,
    const AttentionCapability& plan, int slot, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, const TokenKvPolicy& kv) {
    store_standard_attention_kv_paged(owner_layout, plan, slot, k, v, kv);
    dispatch_standard_attention_paged(attention, owner_layout, plan, slot, q, kv);
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
    __nv_bfloat16* q = prepare_cuda_token_attention_gate(*this, attention, qkv.query);
    __nv_bfloat16* k = qkv.key;
    __nv_bfloat16* v = qkv.value;

    prepare_cuda_token_attention_qk(
        *this, attention, q, k, kv.paged(), kv.rope_position);

    const AttentionCapability plan = token_attention_plan(attention, owner_layout, kv);
    if (kv.paged()) {
        const int slot = kv.paged_kv->attention_slot(resolved_owner.model_layer);
        if (slot < 0) throw std::logic_error("attention layer has no page slot");
        store_and_attend_token_paged(attention, owner_layout, plan, slot, q, k, v, kv);
    } else {
        store_and_attend_token_contiguous(attention, owner, plan, q, k, v);
    }

    apply_cuda_token_attention_gate(*this, attention);
    project_standard_attention_output(attention, semantics);
}

}
