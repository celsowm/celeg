#include "attention_layer_support.hpp"

#include <stdexcept>

namespace celeg {

CudaAttentionOwner resolve_cuda_attention_owner(
    AttentionLayer& attention,
    int layer_index,
    std::vector<Layer>& layers) {
    const int model_layer = attention.kv_owner_layer >= 0
        ? attention.kv_owner_layer : layer_index;
    AttentionLayer* owner = attention.kv_owner_layer >= 0
        ? as_attention(layers.at(static_cast<size_t>(model_layer)))
        : &attention;
    if (!owner) {
        throw std::logic_error("CUDA shared KV owner is not attention");
    }
    return {.layer = owner, .model_layer = model_layer};
}

CudaQkvProjectionView make_cuda_qkv_projection_view(
    AttentionLayer& attention,
    __nv_bfloat16* storage) {
    if (!storage || !attention.query) {
        throw std::logic_error("CUDA attention QKV projection storage is unavailable");
    }
    const int query_projection_width = attention.query->rows;
    __nv_bfloat16* key = storage + query_projection_width;
    return {
        .query = storage,
        .key = key,
        .value = key + attention.layout.key_value_width(),
        .query_projection_width = query_projection_width};
}

const __nv_bfloat16* require_cuda_factorized_latent_bindings(
    const AttentionLayer& attention) {
    const auto* latent = attention.layout.latent_state();
    if (!latent || !latent->factorized_projection() ||
        !attention.latent_query_projection ||
        !attention.latent_query_expansion ||
        !attention.latent_query_norm ||
        !attention.latent_key_projection ||
        !attention.latent_key_norm ||
        !attention.latent_expansion ||
        !attention.gate || !attention.out) {
        throw std::logic_error(
            "CUDA factorized latent attention has incomplete bindings");
    }

    const auto* expansion =
        std::get_if<Bf16LinearStorage>(&attention.latent_expansion->storage);
    if (!expansion || !expansion->data) {
        throw std::logic_error(
            "CUDA factorized latent attention requires BF16 latent expansion storage");
    }
    return expansion->data;
}

Bf16KvView cuda_bf16_kv_view(AttentionLayer& owner) {
    return {
        .keys = owner.key_cache_bf16(),
        .values = owner.value_cache_bf16()};
}

Int8KvView cuda_int8_kv_view(AttentionLayer& owner) {
    return {
        .keys = owner.key_cache_int8_ptr(),
        .values = owner.value_cache_int8_ptr(),
        .key_scales = owner.key_cache_scales_ptr(),
        .value_scales = owner.value_cache_scales_ptr()};
}

}
