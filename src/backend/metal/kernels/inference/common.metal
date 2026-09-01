#include <metal_stdlib>

using namespace metal;

float celeg_bf16_to_float(ushort bits) {
    return as_type<float>(static_cast<uint>(bits) << 16);
}

float celeg_q4_0_value(device const uchar* block, uint column) {
    const uchar packed = block[2 + (column & 15)];
    const uint value = (column & 16) == 0 ? packed & 0x0f : packed >> 4;
    return static_cast<float>(value) - 8.0f;
}

float celeg_q5k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[48 + (sub >> 1) * 32 + within];
    const uint low = (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
    return static_cast<float>(low | (((block[16 + within] >> sub) & 1) << 4));
}

void celeg_q5k_scale_min(device const uchar* scales, uint index,
                         thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

float celeg_half_to_float(ushort bits) {
    return static_cast<float>(as_type<half>(bits));
}

kernel void celeg_embedding(device const float* table [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& token [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = table[static_cast<size_t>(token) * width + index];
}

void celeg_q4k_scale_min(device const uchar* scales, uint index,
                         thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

uint celeg_q4k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[16 + (sub >> 1) * 32 + within];
    return (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
}

int celeg_q6k_value(device const uchar* block, uint column) {
    const uint half_index = column >> 7;
    const uint index = column & 127;
    const uint lane = index & 31;
    const uint group = index >> 5;
    const device uchar* ql = block + half_index * 64;
    const device uchar* qh = block + 128 + half_index * 32;
    if (group == 0) return (ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4);
    if (group == 1) return (ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4);
    if (group == 2) return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

/**
 * @brief Widest head dimension the attention core keeps in registers.
 *
 * Each lane of a simdgroup owns every 32nd dimension of the head, so the
 * per-lane query and accumulator vectors need `head_dim / 32` slots. Eight
 * slots cover every head dimension the supported architectures use; the host
 * rejects wider heads rather than silently truncating them.
 */
constant uint kCelegAttentionSlots = 8;

/// @brief Bias-free score policy for ordinary causal and sliding-window heads.
struct CelegAttentionNoBias {
    float value(uint, uint, uint) const { return 0.0f; }
};

/// @brief Linear distance penalty applied by ALiBi heads.
struct CelegAttentionAlibiBias {
    constant float* slopes;

    float value(uint head, uint query_position, uint position) const {
        return -slopes[head] * static_cast<float>(
            abs(static_cast<int>(query_position) - static_cast<int>(position)));
    }
};

/// @brief Geometry of the one query row a threadgroup attends for.
struct CelegAttentionSpan {
    size_t query_base;
    size_t output_base;
    uint head;
    uint key_head;
    uint key_heads;
    uint head_dim;
    uint sequence_length;
    uint start;
    uint query_position;
    uint page_tokens;
    float scale;
};

/**
 * @brief Attention for a single query row over an unbounded key/value history.
 *
 * One threadgroup serves one `(row, head)` pair. The simdgroups partition the
 * key positions and each keeps a running `(maximum, denominator, accumulator)`
 * triple that is rescaled whenever a larger score appears, so no score array is
 * materialized, `exp` is evaluated once per position rather than once per
 * position and dimension, and the sequence length is bounded only by the KV
 * cache. Simdgroup 0 merges the partial triples through @p shared, which must
 * provide `2 * simdgroups + simdgroups * head_dim` floats.
 */
template <typename Bias>
void celeg_attention_span(device const float* query,
                          device const float* key_cache,
                          device const float* value_cache,
                          device float* output,
                          CelegAttentionSpan span,
                          Bias bias,
                          threadgroup float* shared,
                          uint lane,
                          uint simd,
                          uint simd_count) {
    const uint head_dim = span.head_dim;
    const size_t key_width = static_cast<size_t>(span.key_heads) * head_dim;
    const size_t key_offset = static_cast<size_t>(span.key_head) * head_dim;

    float query_values[kCelegAttentionSlots];
    float accumulator[kCelegAttentionSlots];
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        query_values[slot] = dimension < head_dim ? query[span.query_base + dimension] : 0.0f;
        accumulator[slot] = 0.0f;
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint position = span.start + simd; position < span.sequence_length;
         position += simd_count) {
        const size_t key_base =
            (static_cast<size_t>(position / span.page_tokens) * span.page_tokens +
             position % span.page_tokens) * key_width + key_offset;
        float partial = 0.0f;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint dimension = lane + slot * 32u;
            if (dimension < head_dim) {
                partial += query_values[slot] * key_cache[key_base + dimension];
            }
        }
        const float score = simd_sum(partial) * span.scale +
            bias.value(span.head, span.query_position, position);
        const float updated = max(maximum, score);
        const float correction = exp(maximum - updated);
        const float weight = exp(score - updated);
        denominator = denominator * correction + weight;
        for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
            const uint dimension = lane + slot * 32u;
            if (dimension < head_dim) {
                accumulator[slot] = accumulator[slot] * correction +
                    weight * value_cache[key_base + dimension];
            }
        }
        maximum = updated;
    }

    threadgroup float* shared_maximum = shared;
    threadgroup float* shared_denominator = shared + simd_count;
    threadgroup float* shared_accumulator = shared + 2u * simd_count;
    if (lane == 0) {
        shared_maximum[simd] = maximum;
        shared_denominator[simd] = denominator;
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        if (dimension < head_dim) {
            shared_accumulator[simd * head_dim + dimension] = accumulator[slot];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd != 0) return;

    float global_maximum = -INFINITY;
    for (uint group = 0; group < simd_count; ++group) {
        global_maximum = max(global_maximum, shared_maximum[group]);
    }
    float total = 0.0f;
    for (uint group = 0; group < simd_count; ++group) {
        total += shared_denominator[group] * exp(shared_maximum[group] - global_maximum);
    }
    for (uint slot = 0; slot < kCelegAttentionSlots; ++slot) {
        const uint dimension = lane + slot * 32u;
        if (dimension >= head_dim) continue;
        float value = 0.0f;
        for (uint group = 0; group < simd_count; ++group) {
            value += shared_accumulator[group * head_dim + dimension] *
                exp(shared_maximum[group] - global_maximum);
        }
        output[span.output_base + dimension] = value / total;
    }
}

/// @brief Builds the span a decode threadgroup attends for.
CelegAttentionSpan celeg_attention_decode_span(uint head, uint query_heads, uint key_heads,
                                               uint head_dim, uint sequence_length,
                                               uint start, uint page_tokens, float scale) {
    CelegAttentionSpan span;
    span.query_base = static_cast<size_t>(head) * head_dim;
    span.output_base = span.query_base;
    span.head = head;
    span.key_head = head / (query_heads / key_heads);
    span.key_heads = key_heads;
    span.head_dim = head_dim;
    span.sequence_length = sequence_length;
    span.start = start;
    span.query_position = sequence_length - 1;
    span.page_tokens = page_tokens;
    span.scale = scale;
    return span;
}

/// @brief Builds the span a batched-prefill threadgroup attends for.
CelegAttentionSpan celeg_attention_batch_span(uint head, uint row, uint base_position,
                                              uint query_heads, uint key_heads,
                                              uint head_dim, uint start_window,
                                              uint page_tokens, float scale) {
    const uint query_position = base_position + row;
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim;
    CelegAttentionSpan span;
    span.query_base = static_cast<size_t>(row) * query_width +
        static_cast<size_t>(head) * head_dim;
    span.output_base = span.query_base;
    span.head = head;
    span.key_head = head / (query_heads / key_heads);
    span.key_heads = key_heads;
    span.head_dim = head_dim;
    span.sequence_length = query_position + 1;
    span.start = start_window;
    span.query_position = query_position;
    span.page_tokens = page_tokens;
    span.scale = scale;
    return span;
}

kernel void celeg_attention_sliding(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    constant uint& window_size [[buffer(10)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads) return;
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_decode_span(grid.x, query_heads, key_heads,
                                                     head_dim, sequence_length, start,
                                                     page_tokens, scale),
                         CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}

kernel void celeg_attention_batch_sliding(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& base_position [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& page_tokens [[buffer(10)]],
    constant uint& window_size [[buffer(11)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads || grid.y >= rows) return;
    const uint sequence_length = base_position + grid.y + 1;
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_batch_span(grid.x, grid.y, base_position,
                                                    query_heads, key_heads, head_dim,
                                                    start, page_tokens, scale),
                         CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}

kernel void celeg_attention_alibi(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& key_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& page_tokens [[buffer(9)]],
    constant uint& window_size [[buffer(10)]],
    constant float* slopes [[buffer(11)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads) return;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_decode_span(grid.x, query_heads, key_heads,
                                                     head_dim, sequence_length, start,
                                                     page_tokens, scale),
                         CelegAttentionAlibiBias{slopes}, shared, lane, simd, simd_count);
}

kernel void celeg_attention_batch_alibi(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& base_position [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& key_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant uint& page_tokens [[buffer(10)]],
    constant uint& window_size [[buffer(11)]],
    constant float* slopes [[buffer(12)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads || grid.y >= rows) return;
    const uint sequence_length = base_position + grid.y + 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_batch_span(grid.x, grid.y, base_position,
                                                    query_heads, key_heads, head_dim,
                                                    start, page_tokens, scale),
                         CelegAttentionAlibiBias{slopes}, shared, lane, simd, simd_count);
}
