#include "detail/compiled_model.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {

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

}
