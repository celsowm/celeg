#include "detail/compiled_model.hpp"
#include "attention_contiguous_dispatch.hpp"
#include "attention_kv_store.hpp"
#include "attention_layer_support.hpp"
#include "attention_output_gate.hpp"
#include "attention_projection.hpp"
#include "attention_qk_prepare.hpp"
#include "attention_token_latent.hpp"
#include "attention_token_qk.hpp"
#include "backend/cuda/paged_kv.hpp"

#include <stdexcept>

namespace celeg {

void CudaCompiledModel::run_token_latent_attention_paged(
    AttentionLayer& attention, const CompiledLayerProgram& semantics,
    int layer_index, const TokenKvPolicy& kv) {
    require_cuda_token_latent_attention_paged(*this, attention);
    const AttentionSpec& layout = attention.layout;

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
    execute_cuda_token_latent_attention_paged(*this, attention, layer_index, kv);
    project_latent_attention_output(attention, semantics);
}

void CudaCompiledModel::run_token_attention(
    AttentionLayer& attention, const CompiledLayerProgram& semantics,
    int layer_index, const TokenKvPolicy& kv) {
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
        store_standard_attention_kv_paged(owner_layout, plan, slot, k, v, kv);
        dispatch_standard_attention_paged(attention, owner_layout, plan, slot, q, kv);
    } else {
        store_standard_attention_kv_contiguous(attention, owner, plan, k, v);
        dispatch_cuda_standard_attention_contiguous(*this, attention, owner, plan, q);
    }

    apply_cuda_token_attention_gate(*this, attention);
    project_standard_attention_output(attention, semantics);
}

}
