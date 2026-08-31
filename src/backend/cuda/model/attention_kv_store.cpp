#include "detail/compiled_model.hpp"
#include "attention_kv_store.hpp"
#include "backend/cuda/paged_kv.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {
namespace {

int latent_rotary_width(const AttentionLayer& attention) {
    const auto& latent = *attention.layout.latent_state();
    return attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0
        ? latent.rope_head_dim : 0;
}

const __nv_bfloat16* latent_key_rope_for_store(
    CudaCompiledModel& model, const AttentionLayer& attention) {
    return latent_rotary_width(attention) != 0
        ? model.workspace_.latent_key_rope_.data() : nullptr;
}

} // namespace

void CudaCompiledModel::store_standard_attention_kv_contiguous(
    AttentionLayer& attention, AttentionLayer& owner,
    const AttentionCapability& plan, __nv_bfloat16* k, __nv_bfloat16* v) {
    if (!attention.key || !attention.value) return;

    const AttentionSpec& owner_layout = owner.layout;
    const bool int8_kv = plan.kv_format == KvCacheMode::Int8;
    switch (plan.position_source) {
    case AttentionPositionSource::HostScalar:
        if (int8_kv) {
            launch_store_kv_int8(
                k, v, owner.key_cache_int8_ptr(), owner.value_cache_int8_ptr(),
                owner.key_cache_scales_ptr(), owner.value_cache_scales_ptr(),
                session_.position_, owner_layout.key_value_heads,
                owner_layout.head_dim, stream_.get());
        } else {
            launch_store_kv(
                k, v, owner.key_cache_bf16(), owner.value_cache_bf16(),
                session_.position_, owner_layout.key_value_width(), stream_.get());
        }
        return;

    case AttentionPositionSource::DeviceCounter:
        if (int8_kv) {
            launch_store_kv_int8_device(
                k, v, owner.key_cache_int8_ptr(), owner.value_cache_int8_ptr(),
                owner.key_cache_scales_ptr(), owner.value_cache_scales_ptr(),
                position_device_.data(), owner_layout.key_value_heads,
                owner_layout.head_dim, stream_.get());
        } else {
            launch_store_kv_device(
                k, v, owner.key_cache_bf16(), owner.value_cache_bf16(),
                position_device_.data(), owner_layout.key_value_width(), stream_.get());
        }
        return;
    }

    throw std::logic_error("unsupported CUDA attention KV position source");
}

void CudaCompiledModel::store_standard_attention_kv_paged(
    const AttentionSpec& owner_layout, const AttentionCapability& plan,
    int slot, __nv_bfloat16* k, __nv_bfloat16* v,
    const TokenKvPolicy& kv) {
    PhysicalPagedKvCache& paged_kv = *kv.paged_kv;
    if (plan.kv_format == KvCacheMode::Int8) {
        launch_store_kv_int8_paged_batch(
            k, v, paged_kv.key_int8(), paged_kv.value_int8(),
            paged_kv.key_scales(), paged_kv.value_scales(),
            kv.device_page_table, kv.page_table_stride, position_device_.data(),
            1, slot, paged_kv.page_tokens(), paged_kv.page_vector_elements(),
            paged_kv.layer_vector_offset(slot), paged_kv.page_scale_elements(),
            paged_kv.layer_scale_offset(slot), owner_layout.key_value_heads,
            owner_layout.head_dim, stream_.get());
        return;
    }

    launch_store_kv_paged_batch(
        k, v, paged_kv.key_bf16(), paged_kv.value_bf16(),
        kv.device_page_table, kv.page_table_stride, position_device_.data(),
        1, slot, paged_kv.page_tokens(), paged_kv.page_vector_elements(),
        paged_kv.layer_vector_offset(slot), owner_layout.key_value_heads,
        owner_layout.head_dim, stream_.get());
}

void store_cuda_latent_kv_contiguous(
    CudaCompiledModel& model, AttentionLayer& attention, AttentionLayer& owner) {
    if (!attention.latent_key || !attention.latent_value) return;

    const auto& latent = *attention.layout.latent_state();
    const int rotary_width = latent_rotary_width(attention);
    launch_store_latent_device(
        model.workspace_.latent_key_.data(), model.workspace_.latent_value_.data(),
        latent_key_rope_for_store(model, attention),
        owner.latent_key_cache_ptr(), owner.latent_value_cache_ptr(),
        owner.latent_key_rope_cache_ptr(), model.position_device_.data(),
        latent.latent_rank, rotary_width, model.stream_.get());
}

void store_cuda_latent_kv_paged(
    CudaCompiledModel& model, AttentionLayer& attention,
    PhysicalPagedKvCache& paged_kv, int slot,
    const uint32_t* device_page_table, int page_table_stride) {
    if (slot < 0) {
        throw std::logic_error("latent attention has no page slot");
    }
    if (!attention.latent_key || !attention.latent_value) return;

    const auto& latent = *attention.layout.latent_state();
    const int rotary_width = latent_rotary_width(attention);
    launch_store_latent_paged_batch(
        model.workspace_.latent_key_.data(), model.workspace_.latent_value_.data(),
        latent_key_rope_for_store(model, attention),
        paged_kv.key_bf16(), paged_kv.value_bf16(), device_page_table,
        page_table_stride, model.position_device_.data(), 1, slot,
        paged_kv.page_tokens(), paged_kv.page_vector_elements(),
        paged_kv.layer_vector_offset(slot), latent.latent_rank,
        rotary_width, model.stream_.get());
}

}
