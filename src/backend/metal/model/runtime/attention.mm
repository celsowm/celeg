#include "detail.hpp"

#include <algorithm>
#include <cmath>

namespace celeg {

void MetalModel::Impl::encode_attention(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    encode_matvec(encoder, layer.query, normed, query_buffer);
    encode_matvec(encoder, layer.key, normed, key_buffer);
    encode_matvec(encoder, layer.value, normed, value_buffer);
    const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
    const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
    const uint32_t position_value = static_cast<uint32_t>(position);
    const float query_scale = layer.query_scale /
        (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
    set_buffer(encoder, query_buffer, 0);
    set_buffer(encoder, layer.query_norm, 1);
    set_buffer(encoder, key_buffer, 2);
    set_buffer(encoder, layer.key_norm, 3);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 4);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 5);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 6);
    set_bytes(encoder, &position_value, sizeof(position_value), 7);
    set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 8);
    set_bytes(encoder, &query_scale, sizeof(query_scale), 9);
    set_bytes(encoder, &layer.query_norm_epsilon,
              sizeof(layer.query_norm_epsilon), 10);
    set_bytes(encoder, &layer.key_norm_epsilon,
              sizeof(layer.key_norm_epsilon), 11);
    dispatch(encoder, "celeg_qk_norm_rope",
             std::max(query_heads, key_heads));
    const uint32_t kv_width = key_heads * head_dim;
    set_buffer(encoder, key_buffer, 0);
    set_buffer(encoder, value_buffer, 1);
    set_buffer(encoder, layer.key_cache, 2);
    set_buffer(encoder, layer.value_cache, 3);
    set_bytes(encoder, &position_value, sizeof(position_value), 4);
    set_bytes(encoder, &kv_width, sizeof(kv_width), 5);
    const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
    set_bytes(encoder, &page_tokens, sizeof(page_tokens), 6);
    dispatch(encoder, "celeg_store_kv", kv_width);
    const float attention_scale = 1.0f /
        std::sqrt(static_cast<float>(layer.head_dim));
    set_buffer(encoder, query_buffer, 0);
    set_buffer(encoder, layer.key_cache, 1);
    set_buffer(encoder, layer.value_cache, 2);
    set_buffer(encoder, operation, 3);
    const uint32_t sequence_length = position_value + 1;
    set_bytes(encoder, &sequence_length, sizeof(sequence_length), 4);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 6);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
    set_bytes(encoder, &attention_scale, sizeof(attention_scale), 8);
    set_bytes(encoder, &page_tokens, sizeof(page_tokens), 9);
    dispatch(encoder, "celeg_attention", query_heads * head_dim);
    encode_matvec(encoder, layer.attention_out, operation, hidden);
}

}

