uint celeg_relative_position_bucket(int query_position,
                                    int key_position,
                                    uint bucket_count,
                                    uint max_distance,
                                    uint bidirectional) {
    const int relative_position = key_position - query_position;
    const uint directional_buckets = bidirectional != 0 ? bucket_count / 2 : bucket_count;
    const bool positive = bidirectional != 0 && relative_position > 0;
    const uint distance = bidirectional != 0
        ? static_cast<uint>(abs(relative_position))
        : static_cast<uint>(max(-relative_position, 0));
    const uint max_exact = directional_buckets / 2;
    uint bucket = 0;
    if (distance < max_exact) {
        bucket = distance;
    } else {
        const float denominator = log(
            static_cast<float>(max(max_distance, max_exact + 1)) /
            static_cast<float>(max(max_exact, 1u)));
        const float logarithmic = denominator == 0.0f ? 0.0f : log(
            static_cast<float>(max(distance, max_exact)) /
            static_cast<float>(max(max_exact, 1u))) / denominator;
        bucket = max_exact + static_cast<uint>(
            logarithmic * static_cast<float>(directional_buckets - max_exact));
        bucket = min(bucket, directional_buckets - 1);
    }
    if (positive) bucket += directional_buckets;
    return bucket;
}

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
