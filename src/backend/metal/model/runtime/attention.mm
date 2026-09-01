#include "detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace celeg {

namespace {

uint32_t attention_window_size(const CompiledAttentionProgram& attention) {
    if (const auto* sliding =
            std::get_if<SlidingWindowPattern>(&attention.semantics.pattern)) {
        return static_cast<uint32_t>(sliding->window);
    }
    return 0;
}

const AlibiBiasSpec* attention_alibi(const CompiledAttentionProgram& attention) {
    return std::get_if<AlibiBiasSpec>(&attention.semantics.bias);
}

const RelativePositionBiasSpec* attention_relative_bias(
    const CompiledAttentionProgram& attention) {
    return std::get_if<RelativePositionBiasSpec>(&attention.semantics.bias);
}

bool no_position_encoding(const CompiledAttentionProgram& attention) {
    return std::holds_alternative<NoPositionEncodingSpec>(attention.semantics.position);
}

bool split_half_rope(const CompiledAttentionProgram& attention) {
    const RopePositionSpec* rope = attention.semantics.rope_position();
    return rope && rope->pairing == RopePairingKind::SplitHalf;
}

}

void MetalModel::Impl::encode_attention(
    id<MTLComputeCommandEncoder> encoder, Layer& layer,
    const CompiledAttentionProgram& attention,
    const std::array<int32_t, 3>* rope_position) {
    const AlibiBiasSpec* alibi = attention_alibi(attention);
    const RelativePositionBiasSpec* relative = attention_relative_bias(attention);
    const MultiAxisRopeSpec* multi = attention.semantics.multi_axis_position();
    const bool no_position = no_position_encoding(attention);
    if (alibi && !layer.alibi_slopes) {
        layer.alibi_slopes = buffer(alibi->slopes);
    }
    if (relative && !layer.relative_bias) {
        const int layer_index = static_cast<int>(&layer - layers.data());
        layer.relative_bias = load_vector(
            TensorRole::AttentionRelativePositionBias, layer_index,
            layer.query_heads * relative->bucket_count);
    }

    encode_matvec(encoder, layer.query, normed, query_buffer);
    encode_matvec(encoder, layer.key, normed, key_buffer);
    encode_matvec(encoder, layer.value, normed, value_buffer);
    const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
    const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
    const uint32_t position_value = static_cast<uint32_t>(position);
    const float query_scale = layer.query_scale /
        (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
    const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
    set_buffer(encoder, query_buffer, 0);
    set_buffer(encoder, layer.query_norm, 1);
    set_buffer(encoder, key_buffer, 2);
    set_buffer(encoder, layer.key_norm, 3);
    set_buffer(encoder, value_buffer, 4);
    set_buffer(encoder, layer.key_cache, 5);
    set_buffer(encoder, layer.value_cache, 6);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 7);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 8);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 9);
    if (no_position) {
        set_bytes(encoder, &position_value, sizeof(position_value), 10);
        set_bytes(encoder, &query_scale, sizeof(query_scale), 11);
        set_bytes(encoder, &layer.query_norm_epsilon,
                  sizeof(layer.query_norm_epsilon), 12);
        set_bytes(encoder, &layer.key_norm_epsilon,
                  sizeof(layer.key_norm_epsilon), 13);
        set_bytes(encoder, &page_tokens, sizeof(page_tokens), 14);
        dispatch(encoder, "celeg_qk_norm_store_kv",
                 std::max(query_heads, key_heads));
    } else if (multi) {
        const std::array<int32_t, 3>& resolved_position =
            rope_position ? *rope_position : next_rope_position;
        const std::array<uint32_t, 3> sections{
            static_cast<uint32_t>(multi->sections[0]),
            static_cast<uint32_t>(multi->sections[1]),
            static_cast<uint32_t>(multi->sections[2])};
        set_bytes(encoder, &position_value, sizeof(position_value), 10);
        set_bytes(encoder, resolved_position.data(), sizeof(resolved_position), 11);
        set_bytes(encoder, sections.data(), sizeof(sections), 12);
        set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 13);
        set_bytes(encoder, &query_scale, sizeof(query_scale), 14);
        set_bytes(encoder, &layer.query_norm_epsilon,
                  sizeof(layer.query_norm_epsilon), 15);
        set_bytes(encoder, &layer.key_norm_epsilon,
                  sizeof(layer.key_norm_epsilon), 16);
        set_bytes(encoder, &page_tokens, sizeof(page_tokens), 17);
        dispatch(encoder, "celeg_qk_norm_mrope_store_kv",
                 std::max(query_heads, key_heads));
    } else {
        set_bytes(encoder, &position_value, sizeof(position_value), 10);
        set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 11);
        set_bytes(encoder, &query_scale, sizeof(query_scale), 12);
        set_bytes(encoder, &layer.query_norm_epsilon,
                  sizeof(layer.query_norm_epsilon), 13);
        set_bytes(encoder, &layer.key_norm_epsilon,
                  sizeof(layer.key_norm_epsilon), 14);
        set_bytes(encoder, &page_tokens, sizeof(page_tokens), 15);
        dispatch(encoder,
                 split_half_rope(attention)
                     ? "celeg_qk_norm_rope_store_kv_split"
                     : "celeg_qk_norm_rope_store_kv",
                 std::max(query_heads, key_heads));
    }

    const float attention_scale = 1.0f /
        std::sqrt(static_cast<float>(layer.head_dim));
    const uint32_t sequence_length = position_value + 1;
    const uint32_t window_size = attention_window_size(attention);
    set_buffer(encoder, query_buffer, 0);
    set_buffer(encoder, layer.key_cache, 1);
    set_buffer(encoder, layer.value_cache, 2);
    set_buffer(encoder, operation, 3);
    set_bytes(encoder, &sequence_length, sizeof(sequence_length), 4);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 6);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
    set_bytes(encoder, &attention_scale, sizeof(attention_scale), 8);
    set_bytes(encoder, &page_tokens, sizeof(page_tokens), 9);
    if (relative) {
        const uint32_t bucket_count = static_cast<uint32_t>(relative->bucket_count);
        const uint32_t max_distance = static_cast<uint32_t>(relative->max_distance);
        const uint32_t bidirectional = relative->bidirectional ? 1u : 0u;
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        set_buffer(encoder, layer.relative_bias, 11);
        set_bytes(encoder, &bucket_count, sizeof(bucket_count), 12);
        set_bytes(encoder, &max_distance, sizeof(max_distance), 13);
        set_bytes(encoder, &bidirectional, sizeof(bidirectional), 14);
        if (sequence_length <= 1024) {
            dispatch_cooperative(
                encoder, "celeg_attention_relative_bias_cooperative", query_heads);
        } else {
            dispatch(encoder, "celeg_attention_relative_bias",
                     query_heads * head_dim);
        }
    } else if (alibi) {
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        set_buffer(encoder, layer.alibi_slopes, 11);
        if (sequence_length <= 1024) {
            dispatch_cooperative(encoder, "celeg_attention_alibi_cooperative",
                                 query_heads);
        } else {
            dispatch(encoder, "celeg_attention_alibi", query_heads * head_dim);
        }
    } else if (window_size > 0) {
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        if (sequence_length <= 1024) {
            dispatch_cooperative(
                encoder, "celeg_attention_sliding_cooperative", query_heads);
        } else {
            dispatch(encoder, "celeg_attention_sliding", query_heads * head_dim);
        }
    } else if (sequence_length <= 1024) {
        dispatch_cooperative(encoder, "celeg_attention_cooperative", query_heads);
    } else {
        dispatch(encoder, "celeg_attention", query_heads * head_dim);
    }
    encode_matvec(encoder, layer.attention_out, operation, hidden);
}

void MetalModel::Impl::encode_attention_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer,
    const CompiledAttentionProgram& attention, uint32_t rows,
    uint32_t base_position) {
    const AlibiBiasSpec* alibi = attention_alibi(attention);
    const RelativePositionBiasSpec* relative = attention_relative_bias(attention);
    const bool no_position = no_position_encoding(attention);
    if (alibi && !layer.alibi_slopes) {
        layer.alibi_slopes = buffer(alibi->slopes);
    }
    if (relative && !layer.relative_bias) {
        const int layer_index = static_cast<int>(&layer - layers.data());
        layer.relative_bias = load_vector(
            TensorRole::AttentionRelativePositionBias, layer_index,
            layer.query_heads * relative->bucket_count);
    }

    encode_matmul(encoder, layer.query, batch_normed, batch_query, rows);
    encode_matmul(encoder, layer.key, batch_normed, batch_key, rows);
    encode_matmul(encoder, layer.value, batch_normed, batch_value, rows);
    const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
    const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
    const uint32_t head_count = std::max(query_heads, key_heads);
    const float query_scale = layer.query_scale /
        (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
    set_buffer(encoder, batch_query, 0);
    set_buffer(encoder, layer.query_norm, 1);
    set_buffer(encoder, batch_key, 2);
    set_buffer(encoder, layer.key_norm, 3);
    set_bytes(encoder, &rows, sizeof(rows), 4);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 6);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
    if (no_position) {
        set_bytes(encoder, &query_scale, sizeof(query_scale), 8);
        set_bytes(encoder, &layer.query_norm_epsilon, sizeof(layer.query_norm_epsilon), 9);
        set_bytes(encoder, &layer.key_norm_epsilon, sizeof(layer.key_norm_epsilon), 10);
        dispatch(encoder, "celeg_qk_norm_batch_no_position",
                 static_cast<NSUInteger>(rows) * head_count);
    } else {
        set_bytes(encoder, &base_position, sizeof(base_position), 8);
        set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 9);
        set_bytes(encoder, &query_scale, sizeof(query_scale), 10);
        set_bytes(encoder, &layer.query_norm_epsilon, sizeof(layer.query_norm_epsilon), 11);
        set_bytes(encoder, &layer.key_norm_epsilon, sizeof(layer.key_norm_epsilon), 12);
        dispatch(encoder,
                 split_half_rope(attention)
                     ? "celeg_qk_norm_rope_batch_split"
                     : "celeg_qk_norm_rope_batch",
                 static_cast<NSUInteger>(rows) * head_count);
    }

    const uint32_t kv_width = key_heads * head_dim;
    const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
    set_buffer(encoder, batch_key, 0);
    set_buffer(encoder, batch_value, 1);
    set_buffer(encoder, layer.key_cache, 2);
    set_buffer(encoder, layer.value_cache, 3);
    set_bytes(encoder, &rows, sizeof(rows), 4);
    set_bytes(encoder, &base_position, sizeof(base_position), 5);
    set_bytes(encoder, &kv_width, sizeof(kv_width), 6);
    set_bytes(encoder, &page_tokens, sizeof(page_tokens), 7);
    dispatch(encoder, "celeg_store_kv_batch",
             static_cast<NSUInteger>(rows) * kv_width);

    const float attention_scale = 1.0f /
        std::sqrt(static_cast<float>(layer.head_dim));
    const uint32_t window_size = attention_window_size(attention);
    set_buffer(encoder, batch_query, 0);
    set_buffer(encoder, layer.key_cache, 1);
    set_buffer(encoder, layer.value_cache, 2);
    set_buffer(encoder, batch_operation, 3);
    set_bytes(encoder, &rows, sizeof(rows), 4);
    set_bytes(encoder, &base_position, sizeof(base_position), 5);
    set_bytes(encoder, &query_heads, sizeof(query_heads), 6);
    set_bytes(encoder, &key_heads, sizeof(key_heads), 7);
    set_bytes(encoder, &head_dim, sizeof(head_dim), 8);
    set_bytes(encoder, &attention_scale, sizeof(attention_scale), 9);
    set_bytes(encoder, &page_tokens, sizeof(page_tokens), 10);
    if (relative) {
        const uint32_t bucket_count = static_cast<uint32_t>(relative->bucket_count);
        const uint32_t max_distance = static_cast<uint32_t>(relative->max_distance);
        const uint32_t bidirectional = relative->bidirectional ? 1u : 0u;
        set_bytes(encoder, &window_size, sizeof(window_size), 11);
        set_buffer(encoder, layer.relative_bias, 12);
        set_bytes(encoder, &bucket_count, sizeof(bucket_count), 13);
        set_bytes(encoder, &max_distance, sizeof(max_distance), 14);
        set_bytes(encoder, &bidirectional, sizeof(bidirectional), 15);
        if (base_position + rows <= 1024) {
            id<MTLComputePipelineState> state =
                pipeline("celeg_attention_batch_relative_bias_cooperative");
            [encoder setComputePipelineState:state];
            [encoder dispatchThreadgroups:MTLSizeMake(query_heads, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++command_dispatches;
        } else {
            dispatch(encoder, "celeg_attention_batch_relative_bias",
                     static_cast<NSUInteger>(rows) * query_heads * head_dim);
        }
    } else if (alibi) {
        set_bytes(encoder, &window_size, sizeof(window_size), 11);
        set_buffer(encoder, layer.alibi_slopes, 12);
        if (base_position + rows <= 1024) {
            id<MTLComputePipelineState> state =
                pipeline("celeg_attention_batch_alibi_cooperative");
            [encoder setComputePipelineState:state];
            [encoder dispatchThreadgroups:MTLSizeMake(query_heads, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++command_dispatches;
        } else {
            dispatch(encoder, "celeg_attention_batch_alibi",
                     static_cast<NSUInteger>(rows) * query_heads * head_dim);
        }
    } else if (window_size > 0) {
        set_bytes(encoder, &window_size, sizeof(window_size), 11);
        if (base_position + rows <= 1024) {
            id<MTLComputePipelineState> state =
                pipeline("celeg_attention_batch_sliding_cooperative");
            [encoder setComputePipelineState:state];
            [encoder dispatchThreadgroups:MTLSizeMake(query_heads, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            ++command_dispatches;
        } else {
            dispatch(encoder, "celeg_attention_batch_sliding",
                     static_cast<NSUInteger>(rows) * query_heads * head_dim);
        }
    } else if (base_position + rows <= 1024) {
        id<MTLComputePipelineState> state = pipeline("celeg_attention_batch_cooperative");
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(query_heads, rows, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        ++command_dispatches;
    } else {
        dispatch(encoder, "celeg_attention_batch",
                 static_cast<NSUInteger>(rows) * query_heads * head_dim);
    }
    encode_matmul(encoder, layer.attention_out, batch_operation, batch_hidden, rows);
}

}
