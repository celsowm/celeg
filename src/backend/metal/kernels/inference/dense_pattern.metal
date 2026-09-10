// Dense future-reading attention semantics for Bidirectional and Prefix-LM.
// This file is assembled after attention_one_exp.metal so production reuses
// the existing batch attention reduction/softmax implementation.

bool celeg_attention_bidirectional_visible(int query_position, int key_position) {
    return query_position >= 0 && key_position >= 0;
}

bool celeg_attention_prefix_lm_visible(int query_position, int key_position,
                                       uint prefix_length) {
    if (query_position < 0 || key_position < 0) return false;
    return static_cast<uint>(query_position) < prefix_length
        ? static_cast<uint>(key_position) < prefix_length
        : key_position <= query_position;
}

uint celeg_attention_prefix_lm_visible_sequence_length(
    uint query_position, uint available_sequence_length, uint prefix_length) {
    if (query_position < prefix_length) {
        return min(available_sequence_length, prefix_length);
    }
    return min(available_sequence_length, query_position + 1u);
}

uint celeg_attention_dense_visible_sequence_length(
    uint query_position, uint available_sequence_length,
    uint pattern_mode, uint prefix_length) {
    // 1 = Bidirectional, 2 = Prefix-LM. Other values preserve causal extent.
    if (pattern_mode == 1u) return available_sequence_length;
    if (pattern_mode == 2u) {
        return celeg_attention_prefix_lm_visible_sequence_length(
            query_position, available_sequence_length, prefix_length);
    }
    return min(available_sequence_length, query_position + 1u);
}

kernel void celeg_attention_dense_pattern_semantics_probe(
    device uint* output [[buffer(0)]],
    constant int& query_position [[buffer(1)]],
    constant int& key_position [[buffer(2)]],
    constant uint& available_sequence_length [[buffer(3)]],
    constant uint& prefix_length [[buffer(4)]]) {
    output[0] = celeg_attention_bidirectional_visible(
        query_position, key_position) ? 1u : 0u;
    output[1] = celeg_attention_prefix_lm_visible(
        query_position, key_position, prefix_length) ? 1u : 0u;
    output[2] = query_position < 0 ? 0u :
        celeg_attention_prefix_lm_visible_sequence_length(
            static_cast<uint>(query_position), available_sequence_length,
            prefix_length);
}

kernel void celeg_attention_batch_dense_pattern(
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
    constant uint& pattern_mode [[buffer(11)]],
    constant uint& prefix_length [[buffer(12)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads || grid.y >= rows) return;

    CelegAttentionSpan span = celeg_attention_batch_span(
        grid.x, grid.y, base_position, query_heads, key_heads,
        head_dim, 0u, page_tokens, scale);
    const uint available_sequence_length = base_position + rows;
    span.sequence_length = celeg_attention_dense_visible_sequence_length(
        span.query_position, available_sequence_length,
        pattern_mode, prefix_length);

    celeg_attention_span_one_exp(
        query, key_cache, value_cache, output, span,
        CelegAttentionNoBias{}, shared, lane, simd, simd_count);
}
