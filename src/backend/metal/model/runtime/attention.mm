#include "detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

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

const OrthogonalizeCurrentValueSpec* attention_output_transform(
    const CompiledAttentionProgram& attention) {
    return std::get_if<OrthogonalizeCurrentValueSpec>(
        &attention.semantics.output_transform);
}

bool no_position_encoding(const CompiledAttentionProgram& attention) {
    return std::holds_alternative<NoPositionEncodingSpec>(attention.semantics.position);
}

/// @brief Simdgroups per attention threadgroup; see @c celeg_attention_span.
constexpr uint32_t kAttentionSimdgroups = 8;

/// @brief Widest head dimension @c celeg_attention_span keeps in registers.
constexpr uint32_t kAttentionMaxHeadDim = 32 * 8;

constexpr NSUInteger kTiledAttentionThreads = 128;
constexpr NSUInteger kTiledAttentionQueries = 32;
constexpr NSUInteger kTiledAttentionSharedFloats = 2400;

bool split_half_rope(const CompiledAttentionProgram& attention) {
    const RopePositionSpec* rope = attention.semantics.rope_position();
    return rope && rope->pairing == RopePairingKind::SplitHalf;
}

bool per_head_norm(const std::optional<NormSpec>& norm) {
    return norm && norm->granularity == NormGranularity::PerHead;
}

uint32_t standard_position_mode(const CompiledAttentionProgram& attention) {
    if (no_position_encoding(attention)) return 0;
    return split_half_rope(attention) ? 1u : 2u;
}

template <typename Layer>
void bind_shared_kv(Layer& layer,
                    const CompiledAttentionProgram& attention,
                    std::vector<Layer>& layers,
                    const CompiledModelProgram& program) {
    const auto* consumer =
        std::get_if<SharedKvConsumer>(&attention.semantics.kv_sharing);
    if (!consumer || layer.kv_owner_layer >= 0) return;

    const int layer_index = static_cast<int>(&layer - layers.data());
    for (int owner = 0; owner < layer_index; ++owner) {
        const auto* candidate = std::get_if<CompiledAttentionProgram>(
            &program.layers[static_cast<size_t>(owner)].mixer);
        const auto* publisher = candidate
            ? std::get_if<SharedKvPublisher>(&candidate->semantics.kv_sharing)
            : nullptr;
        if (!publisher || publisher->group != consumer->group) continue;
        layer.kv_owner_layer = owner;
        layer.key_cache = layers[static_cast<size_t>(owner)].key_cache;
        layer.value_cache = layers[static_cast<size_t>(owner)].value_cache;
        return;
    }
    throw std::logic_error("Metal shared KV consumer has no runtime publisher");
}

}

void MetalModel::Impl::encode_attention_span(
    id<MTLComputeCommandEncoder> encoder, std::string_view name,
    uint32_t query_heads, uint32_t rows, uint32_t head_dim) {
    if (head_dim > kAttentionMaxHeadDim) {
        throw std::runtime_error(
            "Metal attention supports head dimensions up to " +
            std::to_string(kAttentionMaxHeadDim) + ", got " + std::to_string(head_dim));
    }
    id<MTLComputePipelineState> state = pipeline(name);
    constexpr NSUInteger threads = 32 * kAttentionSimdgroups;
    if (state.maxTotalThreadsPerThreadgroup < threads) {
        throw std::runtime_error("Metal pipeline cannot run the attention kernel");
    }
    const NSUInteger shared_floats =
        2u * kAttentionSimdgroups + kAttentionSimdgroups * head_dim;
    [encoder setComputePipelineState:state];
    [encoder setThreadgroupMemoryLength:shared_floats * sizeof(float) atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(query_heads, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    record_dispatch(name);
}

void MetalModel::Impl::encode_attention(
    id<MTLComputeCommandEncoder> encoder, Layer& layer,
    const CompiledAttentionProgram& attention,
    const std::array<int32_t, 3>* rope_position) {
    bind_shared_kv(layer, attention, layers, program);
    const bool owns_kv = attention.execution.has_key_value;
    const AlibiBiasSpec* alibi = attention_alibi(attention);
    const RelativePositionBiasSpec* relative = attention_relative_bias(attention);
    const MultiAxisRopeSpec* multi = attention.semantics.multi_axis_position();
    const bool no_position = no_position_encoding(attention);
    const SigmoidAttentionGateSpec* gate = attention.semantics.output_gate
        ? &*attention.semantics.output_gate : nullptr;
    if (alibi && !layer.alibi_slopes) {
        layer.alibi_slopes = buffer(alibi->slopes);
    }
    if (relative && !layer.relative_bias) {
        const int layer_index = static_cast<int>(&layer - layers.data());
        layer.relative_bias = load_vector(
            TensorRole::AttentionRelativePositionBias, layer_index,
            layer.query_heads * relative->bucket_count);
    }

    const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
    const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
    const uint32_t prepared_key_heads = owns_kv ? key_heads : 0;
    const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
    const uint32_t query_width = query_heads * head_dim;
    const uint32_t key_width = key_heads * head_dim;

    if (gate && gate->packed_with_query) {
        encode_matvec(encoder, layer.query, normed, projected);
        const uint32_t rows = 1;
        set_buffer(encoder, projected, 0);
        set_buffer(encoder, query_buffer, 1);
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &query_width, sizeof(query_width), 3);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 4);
        dispatch(encoder, "celeg_extract_attention_query_batch", query_width);
    } else {
        encode_matvec(encoder, layer.query, normed, query_buffer);
        if (gate) {
            encode_matvec(encoder, layer.attention_gate, normed, projected);
        }
    }
    if (owns_kv) {
        encode_matvec(encoder, layer.key, normed, key_buffer);
        encode_matvec(encoder, layer.value, normed, value_buffer);
    }

    const uint32_t position_value = static_cast<uint32_t>(position);
    const float query_scale = layer.query_scale /
        (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
    const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
    const bool fused_per_head = owns_kv &&
        per_head_norm(attention.semantics.query_norm) &&
        per_head_norm(attention.semantics.key_norm);

    if (!fused_per_head) {
        const auto normalize = [&](id<MTLBuffer> data, id<MTLBuffer> weight,
                                   const std::optional<NormSpec>& norm,
                                   uint32_t heads, uint32_t width) {
            if (!norm) return;
            if (norm->granularity == NormGranularity::WholeVector) {
                encode_rmsnorm(encoder, data, weight, data, width, norm->epsilon);
                return;
            }
            set_buffer(encoder, data, 0);
            set_buffer(encoder, weight, 1);
            set_bytes(encoder, &heads, sizeof(heads), 2);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 3);
            set_bytes(encoder, &norm->epsilon, sizeof(norm->epsilon), 4);
            dispatch(encoder, "celeg_head_rmsnorm_inplace", heads);
        };
        normalize(query_buffer, layer.query_norm, attention.semantics.query_norm,
                  query_heads, query_width);
        if (owns_kv) {
            normalize(key_buffer, layer.key_norm, attention.semantics.key_norm,
                      key_heads, key_width);
        }

        if (multi) {
            const std::array<int32_t, 3>& resolved_position =
                rope_position ? *rope_position : next_rope_position;
            const std::array<uint32_t, 3> sections{
                static_cast<uint32_t>(multi->sections[0]),
                static_cast<uint32_t>(multi->sections[1]),
                static_cast<uint32_t>(multi->sections[2])};
            set_buffer(encoder, query_buffer, 0);
            set_buffer(encoder, key_buffer, 1);
            set_buffer(encoder, value_buffer, 2);
            set_buffer(encoder, layer.key_cache, 3);
            set_buffer(encoder, layer.value_cache, 4);
            set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
            set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 6);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
            set_bytes(encoder, &position_value, sizeof(position_value), 8);
            set_bytes(encoder, resolved_position.data(), sizeof(resolved_position), 9);
            set_bytes(encoder, sections.data(), sizeof(sections), 10);
            set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 11);
            set_bytes(encoder, &query_scale, sizeof(query_scale), 12);
            set_bytes(encoder, &page_tokens, sizeof(page_tokens), 13);
            dispatch(encoder, "celeg_qk_mrope_position_store_kv",
                     std::max(query_heads, prepared_key_heads));
        } else {
            const uint32_t position_mode = standard_position_mode(attention);
            set_buffer(encoder, query_buffer, 0);
            set_buffer(encoder, key_buffer, 1);
            set_buffer(encoder, value_buffer, 2);
            set_buffer(encoder, layer.key_cache, 3);
            set_buffer(encoder, layer.value_cache, 4);
            set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
            set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 6);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 7);
            set_bytes(encoder, &position_value, sizeof(position_value), 8);
            set_bytes(encoder, &position_mode, sizeof(position_mode), 9);
            set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 10);
            set_bytes(encoder, &query_scale, sizeof(query_scale), 11);
            set_bytes(encoder, &page_tokens, sizeof(page_tokens), 12);
            dispatch(encoder, "celeg_qk_position_store_kv",
                     std::max(query_heads, prepared_key_heads));
        }
    } else {
        set_buffer(encoder, query_buffer, 0);
        set_buffer(encoder, layer.query_norm, 1);
        set_buffer(encoder, key_buffer, 2);
        set_buffer(encoder, layer.key_norm, 3);
        set_buffer(encoder, value_buffer, 4);
        set_buffer(encoder, layer.key_cache, 5);
        set_buffer(encoder, layer.value_cache, 6);
        set_bytes(encoder, &query_heads, sizeof(query_heads), 7);
        set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 8);
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
                     std::max(query_heads, prepared_key_heads));
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
                     std::max(query_heads, prepared_key_heads));
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
                     std::max(query_heads, prepared_key_heads));
        }
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
    std::string_view attention_kernel = "celeg_attention";
    if (relative) {
        const uint32_t bucket_count = static_cast<uint32_t>(relative->bucket_count);
        const uint32_t max_distance = static_cast<uint32_t>(relative->max_distance);
        const uint32_t bidirectional = relative->bidirectional ? 1u : 0u;
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        set_buffer(encoder, layer.relative_bias, 11);
        set_bytes(encoder, &bucket_count, sizeof(bucket_count), 12);
        set_bytes(encoder, &max_distance, sizeof(max_distance), 13);
        set_bytes(encoder, &bidirectional, sizeof(bidirectional), 14);
        attention_kernel = "celeg_attention_relative_bias";
    } else if (alibi) {
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        set_buffer(encoder, layer.alibi_slopes, 11);
        attention_kernel = "celeg_attention_alibi";
    } else if (window_size > 0) {
        set_bytes(encoder, &window_size, sizeof(window_size), 10);
        attention_kernel = "celeg_attention_sliding";
    }
    encode_attention_span(encoder, attention_kernel, query_heads, 1, head_dim);

    if (const auto* transform = attention_output_transform(attention)) {
        const uint32_t rows = 1;
        set_buffer(encoder, operation, 0);
        if (owns_kv) {
            set_buffer(encoder, value_buffer, 1);
        } else {
            const NSUInteger value_offset =
                static_cast<NSUInteger>(position_value) * key_width * sizeof(float);
            [encoder setBuffer:layer.value_cache offset:value_offset atIndex:1];
        }
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &query_heads, sizeof(query_heads), 3);
        set_bytes(encoder, &key_heads, sizeof(key_heads), 4);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 5);
        set_bytes(encoder, &transform->minimum_norm_squared,
                  sizeof(transform->minimum_norm_squared), 6);
        dispatch(encoder, "celeg_attention_orthogonalize_current_value",
                 query_heads);
    }
    if (gate) {
        const uint32_t head_wise =
            gate->granularity == AttentionGateGranularity::HeadWise ? 1u : 0u;
        const uint32_t packed = gate->packed_with_query ? 1u : 0u;
        set_buffer(encoder, operation, 0);
        set_buffer(encoder, projected, 1);
        set_bytes(encoder, &query_width, sizeof(query_width), 2);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 3);
        set_bytes(encoder, &head_wise, sizeof(head_wise), 4);
        set_bytes(encoder, &packed, sizeof(packed), 5);
        dispatch(encoder, "celeg_attention_output_gate", query_width);
    }
    encode_matvec(encoder, layer.attention_out, operation, hidden);
}

void MetalModel::Impl::encode_attention_batch(
    id<MTLComputeCommandEncoder> encoder, Layer& layer,
    const CompiledAttentionProgram& attention, uint32_t rows,
    uint32_t base_position) {
    bind_shared_kv(layer, attention, layers, program);
    const bool owns_kv = attention.execution.has_key_value;
    const AlibiBiasSpec* alibi = attention_alibi(attention);
    const RelativePositionBiasSpec* relative = attention_relative_bias(attention);
    const MultiAxisRopeSpec* multi = attention.semantics.multi_axis_position();
    const bool no_position = no_position_encoding(attention);
    const SigmoidAttentionGateSpec* gate = attention.semantics.output_gate
        ? &*attention.semantics.output_gate : nullptr;
    if (alibi && !layer.alibi_slopes) {
        layer.alibi_slopes = buffer(alibi->slopes);
    }
    if (relative && !layer.relative_bias) {
        const int layer_index = static_cast<int>(&layer - layers.data());
        layer.relative_bias = load_vector(
            TensorRole::AttentionRelativePositionBias, layer_index,
            layer.query_heads * relative->bucket_count);
    }

    const uint32_t query_heads = static_cast<uint32_t>(layer.query_heads);
    const uint32_t key_heads = static_cast<uint32_t>(layer.key_value_heads);
    const uint32_t prepared_key_heads = owns_kv ? key_heads : 0;
    const uint32_t head_dim = static_cast<uint32_t>(layer.head_dim);
    const uint32_t query_width = query_heads * head_dim;
    const uint32_t key_width = key_heads * head_dim;

    if (gate && gate->packed_with_query) {
        encode_matmul(encoder, layer.query, batch_normed, batch_projected, rows);
        set_buffer(encoder, batch_projected, 0);
        set_buffer(encoder, batch_query, 1);
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &query_width, sizeof(query_width), 3);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 4);
        dispatch(encoder, "celeg_extract_attention_query_batch",
                 static_cast<NSUInteger>(rows) * query_width);
    } else {
        encode_matmul(encoder, layer.query, batch_normed, batch_query, rows);
        if (gate) {
            encode_matmul(encoder, layer.attention_gate, batch_normed,
                          batch_projected, rows);
        }
    }
    if (owns_kv) {
        encode_matmul(encoder, layer.key, batch_normed, batch_key, rows);
        encode_matmul(encoder, layer.value, batch_normed, batch_value, rows);
    }

    const uint32_t head_count = std::max(query_heads, prepared_key_heads);
    const float query_scale = layer.query_scale /
        (1.0f / std::sqrt(static_cast<float>(layer.head_dim)));
    const bool fused_per_head = owns_kv && !multi &&
        per_head_norm(attention.semantics.query_norm) &&
        per_head_norm(attention.semantics.key_norm);
    const bool qk_publishes_kv =
        fused_per_head && !no_position && split_half_rope(attention);

    if (!fused_per_head) {
        const auto normalize = [&](id<MTLBuffer> data, id<MTLBuffer> weight,
                                   const std::optional<NormSpec>& norm,
                                   uint32_t heads, uint32_t width) {
            if (!norm) return;
            if (norm->granularity == NormGranularity::WholeVector) {
                encode_rmsnorm_batch(encoder, data, weight, data,
                                     rows, width, norm->epsilon);
                return;
            }
            set_buffer(encoder, data, 0);
            set_buffer(encoder, weight, 1);
            set_bytes(encoder, &rows, sizeof(rows), 2);
            set_bytes(encoder, &heads, sizeof(heads), 3);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 4);
            set_bytes(encoder, &norm->epsilon, sizeof(norm->epsilon), 5);
            dispatch(encoder, "celeg_head_rmsnorm_batch_inplace",
                     static_cast<NSUInteger>(rows) * heads);
        };
        normalize(batch_query, layer.query_norm, attention.semantics.query_norm,
                  query_heads, query_width);
        if (owns_kv) {
            normalize(batch_key, layer.key_norm, attention.semantics.key_norm,
                      key_heads, key_width);
        }

        if (multi) {
            const std::array<uint32_t, 3> sections{
                static_cast<uint32_t>(multi->sections[0]),
                static_cast<uint32_t>(multi->sections[1]),
                static_cast<uint32_t>(multi->sections[2])};
            set_buffer(encoder, batch_query, 0);
            set_buffer(encoder, batch_key, 1);
            set_bytes(encoder, &rows, sizeof(rows), 2);
            set_bytes(encoder, &query_heads, sizeof(query_heads), 3);
            set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 4);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 5);
            set_buffer(encoder, batch_rope_positions, 6);
            set_bytes(encoder, sections.data(), sizeof(sections), 7);
            set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 8);
            set_bytes(encoder, &query_scale, sizeof(query_scale), 9);
            dispatch(encoder, "celeg_qk_mrope_position_batch",
                     static_cast<NSUInteger>(rows) * head_count);
        } else {
            const uint32_t position_mode = standard_position_mode(attention);
            set_buffer(encoder, batch_query, 0);
            set_buffer(encoder, batch_key, 1);
            set_bytes(encoder, &rows, sizeof(rows), 2);
            set_bytes(encoder, &query_heads, sizeof(query_heads), 3);
            set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 4);
            set_bytes(encoder, &head_dim, sizeof(head_dim), 5);
            set_bytes(encoder, &base_position, sizeof(base_position), 6);
            set_bytes(encoder, &position_mode, sizeof(position_mode), 7);
            set_bytes(encoder, &layer.rope_theta, sizeof(layer.rope_theta), 8);
            set_bytes(encoder, &query_scale, sizeof(query_scale), 9);
            dispatch(encoder, "celeg_qk_position_batch",
                     static_cast<NSUInteger>(rows) * head_count);
        }
    } else {
        set_buffer(encoder, batch_query, 0);
        set_buffer(encoder, layer.query_norm, 1);
        set_buffer(encoder, batch_key, 2);
        set_buffer(encoder, layer.key_norm, 3);
        set_bytes(encoder, &rows, sizeof(rows), 4);
        set_bytes(encoder, &query_heads, sizeof(query_heads), 5);
        set_bytes(encoder, &prepared_key_heads, sizeof(prepared_key_heads), 6);
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
            if (split_half_rope(attention)) {
                set_buffer(encoder, batch_value, 13);
                set_buffer(encoder, layer.key_cache, 14);
                set_buffer(encoder, layer.value_cache, 15);
                dispatch(encoder, "celeg_qk_norm_rope_batch_split_store_kv",
                         static_cast<NSUInteger>(rows) * head_count);
            } else {
                dispatch(encoder, "celeg_qk_norm_rope_batch",
                         static_cast<NSUInteger>(rows) * head_count);
            }
        }
    }

    const uint32_t kv_width = key_heads * head_dim;
    const uint32_t page_tokens = static_cast<uint32_t>(layer.page_tokens);
    if (owns_kv && !qk_publishes_kv) {
        set_buffer(encoder, batch_key, 0);
        set_buffer(encoder, batch_value, 1);
        set_buffer(encoder, layer.key_cache, 2);
        set_buffer(encoder, layer.value_cache, 3);
        set_bytes(encoder, &rows, sizeof(rows), 4);
        set_bytes(encoder, &base_position, sizeof(base_position), 5);
        set_bytes(encoder, &kv_width, sizeof(kv_width), 6);
        id<MTLComputePipelineState> state = pipeline("celeg_store_kv_batch_2d");
        constexpr NSUInteger threads = 256;
        if (state.maxTotalThreadsPerThreadgroup < threads) {
            throw std::runtime_error("Metal pipeline cannot run the batch KV store kernel");
        }
        [encoder setComputePipelineState:state];
        [encoder dispatchThreadgroups:MTLSizeMake(
            (kv_width + threads - 1u) / threads, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        record_dispatch("celeg_store_kv_batch_2d");
    }

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

    bool tiled_encoded = false;
    const bool tiled_candidate =
        options.numerical_policy == MetalNumericalPolicy::Fast &&
        relative == nullptr && alibi == nullptr &&
        window_size == 0u && base_position == 0u && head_dim == 64u &&
        (rows % kTiledAttentionQueries) == 0u;
    if (tiled_candidate) {
        id<MTLComputePipelineState> state =
            pipeline("celeg_attention_tiled_simdgroup");
        const NSUInteger shared_bytes =
            kTiledAttentionSharedFloats * sizeof(float);
        if (state.maxTotalThreadsPerThreadgroup >= kTiledAttentionThreads &&
            state.staticThreadgroupMemoryLength + shared_bytes <=
                device.maxThreadgroupMemoryLength) {
            [encoder setComputePipelineState:state];
            [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
            [encoder dispatchThreadgroups:MTLSizeMake(
                query_heads, rows / kTiledAttentionQueries, 1)
                   threadsPerThreadgroup:MTLSizeMake(kTiledAttentionThreads, 1, 1)];
            record_dispatch("celeg_attention_tiled_simdgroup");
            tiled_encoded = true;
        }
    }

    if (!tiled_encoded) {
        std::string_view attention_kernel = "celeg_attention_batch";
        if (relative) {
            const uint32_t bucket_count = static_cast<uint32_t>(relative->bucket_count);
            const uint32_t max_distance = static_cast<uint32_t>(relative->max_distance);
            const uint32_t bidirectional = relative->bidirectional ? 1u : 0u;
            set_bytes(encoder, &window_size, sizeof(window_size), 11);
            set_buffer(encoder, layer.relative_bias, 12);
            set_bytes(encoder, &bucket_count, sizeof(bucket_count), 13);
            set_bytes(encoder, &max_distance, sizeof(max_distance), 14);
            set_bytes(encoder, &bidirectional, sizeof(bidirectional), 15);
            attention_kernel = "celeg_attention_batch_relative_bias";
        } else if (alibi) {
            set_bytes(encoder, &window_size, sizeof(window_size), 11);
            set_buffer(encoder, layer.alibi_slopes, 12);
            attention_kernel = "celeg_attention_batch_alibi";
        } else if (window_size > 0) {
            set_bytes(encoder, &window_size, sizeof(window_size), 11);
            attention_kernel = "celeg_attention_batch_sliding";
        }
        encode_attention_span(encoder, attention_kernel, query_heads, rows, head_dim);
    }

    if (const auto* transform = attention_output_transform(attention)) {
        set_buffer(encoder, batch_operation, 0);
        if (owns_kv) {
            set_buffer(encoder, batch_value, 1);
        } else {
            const NSUInteger value_offset =
                static_cast<NSUInteger>(base_position) * kv_width * sizeof(float);
            [encoder setBuffer:layer.value_cache offset:value_offset atIndex:1];
        }
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &query_heads, sizeof(query_heads), 3);
        set_bytes(encoder, &key_heads, sizeof(key_heads), 4);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 5);
        set_bytes(encoder, &transform->minimum_norm_squared,
                  sizeof(transform->minimum_norm_squared), 6);
        dispatch(encoder, "celeg_attention_orthogonalize_current_value",
                 static_cast<NSUInteger>(rows) * query_heads);
    }
    if (gate) {
        const uint32_t head_wise =
            gate->granularity == AttentionGateGranularity::HeadWise ? 1u : 0u;
        const uint32_t packed = gate->packed_with_query ? 1u : 0u;
        const uint32_t gate_row_stride = packed != 0
            ? 2 * query_width
            : (head_wise != 0 ? query_heads : query_width);
        set_buffer(encoder, batch_operation, 0);
        set_buffer(encoder, batch_projected, 1);
        set_bytes(encoder, &rows, sizeof(rows), 2);
        set_bytes(encoder, &query_width, sizeof(query_width), 3);
        set_bytes(encoder, &head_dim, sizeof(head_dim), 4);
        set_bytes(encoder, &head_wise, sizeof(head_wise), 5);
        set_bytes(encoder, &packed, sizeof(packed), 6);
        set_bytes(encoder, &gate_row_stride, sizeof(gate_row_stride), 7);
        dispatch(encoder, "celeg_attention_output_gate_batch",
                 static_cast<NSUInteger>(rows) * query_width);
    }
    encode_matmul(encoder, layer.attention_out, batch_operation, batch_hidden, rows);
}

}
