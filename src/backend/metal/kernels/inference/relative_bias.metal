float celeg_relative_position_bias(device const float* values,
                                   uint head,
                                   int query_position,
                                   int key_position,
                                   uint bucket_count,
                                   uint max_distance,
                                   uint bidirectional) {
    const uint bucket = celeg_relative_position_bucket(
        query_position, key_position, bucket_count, max_distance, bidirectional);
    return values[static_cast<size_t>(head) * bucket_count + bucket];
}

/// @brief Bucketed relative-position penalty applied by T5-style heads.
struct CelegAttentionRelativeBias {
    device const float* values;
    uint bucket_count;
    uint max_distance;
    uint bidirectional;

    float value(uint head, uint query_position, uint position) const {
        return celeg_relative_position_bias(
            values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
    }
};

kernel void celeg_attention_relative_bias(
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
    device const float* bias_values [[buffer(11)]],
    constant uint& bucket_count [[buffer(12)]],
    constant uint& max_distance [[buffer(13)]],
    constant uint& bidirectional [[buffer(14)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    if (grid.x >= query_heads) return;
    const CelegAttentionRelativeBias bias{bias_values, bucket_count, max_distance,
                                          bidirectional};
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_decode_span(grid.x, query_heads, key_heads,
                                                     head_dim, sequence_length, start,
                                                     page_tokens, scale),
                         bias, shared, lane, simd, simd_count);
}

kernel void celeg_attention_batch_relative_bias(
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
    device const float* bias_values [[buffer(12)]],
    constant uint& bucket_count [[buffer(13)]],
    constant uint& max_distance [[buffer(14)]],
    constant uint& bidirectional [[buffer(15)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    if (grid.x >= query_heads || grid.y >= rows) return;
    const uint sequence_length = base_position + grid.y + 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    const CelegAttentionRelativeBias bias{bias_values, bucket_count, max_distance,
                                          bidirectional};
    celeg_attention_span(query, key_cache, value_cache, output,
                         celeg_attention_batch_span(grid.x, grid.y, base_position,
                                                    query_heads, key_heads, head_dim,
                                                    start, page_tokens, scale),
                         bias, shared, lane, simd, simd_count);
}
