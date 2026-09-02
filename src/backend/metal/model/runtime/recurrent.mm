#include "detail.hpp"

namespace celeg {

void MetalModel::Impl::encode_gated_delta(id<MTLComputeCommandEncoder> encoder, Layer& layer) {
        const uint32_t conv_kernel = static_cast<uint32_t>(layer.recurrent_conv_kernel);
        const uint32_t key_head_dim = static_cast<uint32_t>(layer.recurrent_key_head_dim);
        const uint32_t value_head_dim = static_cast<uint32_t>(layer.recurrent_value_head_dim);
        const uint32_t key_heads = static_cast<uint32_t>(layer.recurrent_key_heads);
        const uint32_t value_heads = static_cast<uint32_t>(layer.recurrent_value_heads);
        const float epsilon = program.final_norm.epsilon;
        const uint32_t vector_decay = layer.recurrent_vector_decay ? 1 : 0;
        const uint32_t safe_decay = layer.recurrent_safe_decay ? 1 : 0;
        const uint32_t sigmoid_output_gate = layer.recurrent_sigmoid_output_gate ? 1 : 0;
        const uint32_t a_log_needs_exp = layer.recurrent_a_log_needs_exp ? 1 : 0;
        set_buffer(encoder, projected, 0);
        set_buffer(encoder, recurrent_z, 1);
        set_buffer(encoder, recurrent_b, 2);
        set_buffer(encoder, recurrent_a, 3);
        set_buffer(encoder, layer.recurrent_conv_weight, 4);
        set_buffer(encoder, layer.recurrent_dt_bias, 5);
        set_buffer(encoder, layer.recurrent_a_log, 6);
        set_buffer(encoder, layer.recurrent_norm, 7);
        set_buffer(encoder, layer.recurrent_conv_state, 8);
        set_buffer(encoder, layer.recurrent_state, 9);
        set_buffer(encoder, recurrent_output, 10);
        set_bytes(encoder, &conv_kernel, sizeof(conv_kernel), 11);
        set_bytes(encoder, &key_head_dim, sizeof(key_head_dim), 12);
        set_bytes(encoder, &value_head_dim, sizeof(value_head_dim), 13);
        set_bytes(encoder, &key_heads, sizeof(key_heads), 14);
        set_bytes(encoder, &value_heads, sizeof(value_heads), 15);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 16);
        set_bytes(encoder, &vector_decay, sizeof(vector_decay), 17);
        set_bytes(encoder, &safe_decay, sizeof(safe_decay), 18);
        set_bytes(encoder, &layer.recurrent_decay_lower_bound,
                  sizeof(layer.recurrent_decay_lower_bound), 19);
        set_bytes(encoder, &sigmoid_output_gate, sizeof(sigmoid_output_gate), 20);
        set_bytes(encoder, &a_log_needs_exp, sizeof(a_log_needs_exp), 21);
        dispatch(encoder, "celeg_gated_delta", 1);
    }

void MetalModel::Impl::encode_mamba2(id<MTLComputeCommandEncoder> encoder, Layer& layer) {
        const uint32_t inner = static_cast<uint32_t>(layer.recurrent_inner);
        const uint32_t state_size = static_cast<uint32_t>(layer.recurrent_state_size);
        const uint32_t num_heads = static_cast<uint32_t>(layer.recurrent_value_heads);
        const uint32_t head_dim = static_cast<uint32_t>(layer.recurrent_key_head_dim);
        const uint32_t group_count = static_cast<uint32_t>(layer.recurrent_group_count);
        const uint32_t conv_kernel = static_cast<uint32_t>(layer.recurrent_conv_kernel);
        const float epsilon = program.final_norm.epsilon;
        const uint32_t a_log_needs_exp = layer.recurrent_a_log_needs_exp ? 1 : 0;
        set_buffer(encoder, projected, 0);
        set_buffer(encoder, layer.recurrent_conv_weight, 1);
        set_buffer(encoder, layer.recurrent_conv_bias, 2);
        set_buffer(encoder, layer.recurrent_dt_bias, 3);
        set_buffer(encoder, layer.recurrent_a_log, 4);
        set_buffer(encoder, layer.recurrent_d, 5);
        set_buffer(encoder, layer.recurrent_norm, 6);
        set_buffer(encoder, layer.recurrent_conv_state, 7);
        set_buffer(encoder, layer.recurrent_state, 8);
        set_buffer(encoder, recurrent_output, 9);
        set_bytes(encoder, &inner, sizeof(inner), 10);
        set_bytes(encoder, &state_size, sizeof(state_size), 11);
        set_bytes(encoder, &num_heads, sizeof(num_heads), 12);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 13);
        set_bytes(encoder, &group_count, sizeof(group_count), 14);
        set_bytes(encoder, &conv_kernel, sizeof(conv_kernel), 15);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 16);
        set_bytes(encoder, &a_log_needs_exp, sizeof(a_log_needs_exp), 17);
        dispatch(encoder, "celeg_mamba2", 1);
    }

void MetalModel::Impl::encode_short_convolution(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
    encode_matvec(encoder, layer.mixer_in, normed, projected);
    const uint32_t cache_length = static_cast<uint32_t>(layer.cache_length);
    const uint32_t position_value = static_cast<uint32_t>(position);
    const uint32_t cursor = position_value % cache_length;
    set_buffer(encoder, projected, 0);
    set_buffer(encoder, layer.convolution_taps, 1);
    set_buffer(encoder, layer.key_cache, 2);
    set_buffer(encoder, operation, 3);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 4);
    set_bytes(encoder, &cache_length, sizeof(cache_length), 5);
    set_bytes(encoder, &cursor, sizeof(cursor), 6);
    dispatch(encoder, "celeg_shortconv_ring", hidden_width);
    encode_matvec(encoder, layer.mixer_out, operation, hidden);
}

void MetalModel::Impl::encode_short_convolution_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer, uint32_t rows,
    uint32_t base_position) {
    const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
    const uint32_t cache_length = static_cast<uint32_t>(layer.cache_length);
    const uint32_t initial_cursor = base_position % cache_length;
    const uint32_t element_count = rows * hidden_width;
    const uint32_t state_count = std::min(rows, cache_length) * hidden_width;

    set_buffer(encoder, batch_projected, 0);
    set_buffer(encoder, batch_activated, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 3);
    dispatch(encoder, "celeg_shortconv_batch_gate_parallel", element_count);

    set_buffer(encoder, batch_projected, 0);
    set_buffer(encoder, layer.convolution_taps, 1);
    set_buffer(encoder, layer.key_cache, 2);
    set_buffer(encoder, batch_activated, 3);
    set_buffer(encoder, batch_operation, 4);
    set_bytes(encoder, &rows, sizeof(rows), 5);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 6);
    set_bytes(encoder, &cache_length, sizeof(cache_length), 7);
    set_bytes(encoder, &initial_cursor, sizeof(initial_cursor), 8);
    dispatch(encoder, "celeg_shortconv_batch_convolve_parallel", element_count);

    set_buffer(encoder, batch_activated, 0);
    set_buffer(encoder, layer.key_cache, 1);
    set_bytes(encoder, &rows, sizeof(rows), 2);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 3);
    set_bytes(encoder, &cache_length, sizeof(cache_length), 4);
    set_bytes(encoder, &initial_cursor, sizeof(initial_cursor), 5);
    dispatch(encoder, "celeg_shortconv_batch_publish_state_parallel", state_count);

    encode_matmul(encoder, layer.mixer_out, batch_operation, batch_hidden, rows);
}

void MetalModel::Impl::encode_gated_delta_layer(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    const uint32_t key_width = static_cast<uint32_t>(
        layer.recurrent_key_heads * layer.recurrent_key_head_dim);
    const uint32_t value_width = static_cast<uint32_t>(
        layer.recurrent_value_heads * layer.recurrent_value_head_dim);
    if (layer.recurrent_qkv.buffer) {
        encode_matvec(encoder, layer.recurrent_qkv, normed, projected);
    } else {
        encode_matvec(encoder, layer.recurrent_q, normed, projected, 0);
        encode_matvec(encoder, layer.recurrent_k, normed, projected,
                      static_cast<NSUInteger>(key_width) * sizeof(float));
        encode_matvec(encoder, layer.recurrent_v, normed, projected,
                      static_cast<NSUInteger>(2 * key_width) * sizeof(float));
    }
    encode_matvec(encoder, layer.recurrent_z_weight, normed, recurrent_z);
    encode_matvec(encoder, layer.recurrent_b, normed, recurrent_b);
    encode_matvec(encoder, layer.recurrent_a, normed, recurrent_a);
    encode_gated_delta(encoder, layer);
    encode_matvec(encoder, layer.recurrent_out, recurrent_output, hidden);
}

void MetalModel::Impl::encode_mamba2_layer(
    id<MTLComputeCommandEncoder> encoder, Layer& layer) {
    encode_matvec(encoder, layer.recurrent_in, normed, projected);
    encode_mamba2(encoder, layer);
    encode_matvec(encoder, layer.recurrent_out, recurrent_output, hidden);
}

}
