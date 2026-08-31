#include "attention_token_latent.hpp"

#include "attention_kv_store.hpp"
#include "attention_latent_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "backend/cuda/paged_kv.hpp"

#include <stdexcept>

namespace celeg {

void require_cuda_token_latent_attention_paged(
    const CudaCompiledModel& model, const AttentionLayer& attention) {
    if (model.resources_.options().kv_cache_mode == KvCacheMode::Int8) {
        throw std::invalid_argument(
            "CUDA latent attention requires BF16 paged state storage");
    }
    if (!attention.layout.latent_state()) {
        throw std::logic_error("CUDA latent attention has no latent state specification");
    }
    if (attention.layout.output_gate.has_value() ||
        attention.layout.multi_axis_position()) {
        throw std::invalid_argument(
            "CUDA latent attention does not support query gates or M-RoPE yet");
    }
}

void execute_cuda_token_latent_attention_paged(
    CudaCompiledModel& model, AttentionLayer& attention, int layer_index,
    const CudaCompiledModel::TokenKvPolicy& kv) {
    if (!kv.paged() || !kv.paged_kv) {
        throw std::logic_error("CUDA token latent attention requires paged KV state");
    }

    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    const CudaAttentionOwner resolved_owner = resolve_cuda_attention_owner(
        attention, layer_index, model.resources_.layers_);
    const int slot = paged_kv.attention_slot(resolved_owner.model_layer);

    store_cuda_latent_kv_paged(
        model, attention, paged_kv, slot,
        kv.device_page_table, kv.page_table_stride);
    dispatch_cuda_latent_attention_paged(
        model, attention, paged_kv, slot,
        kv.device_page_table, kv.page_table_stride);
}

}
