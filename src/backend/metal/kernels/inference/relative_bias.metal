struct CelegMergeTransition {
    float maximum;
    float denominator;
    float destination_scale;
    float source_scale;
};

float celeg_partial_rescale(float partial_maximum, float global_maximum) {
    return exp(partial_maximum - global_maximum);
}

CelegMergeTransition celeg_merge_pair(
    float destination_maximum, float destination_denominator,
    float source_maximum, float source_denominator) {
    const float maximum = max(destination_maximum, source_maximum);
    const float destination_scale = isfinite(destination_maximum)
        ? celeg_partial_rescale(destination_maximum, maximum) : 0.0f;
    const float source_scale = celeg_partial_rescale(source_maximum, maximum);
    return CelegMergeTransition{
        maximum,
        destination_denominator * destination_scale +
            source_denominator * source_scale,
        destination_scale,
        source_scale};
}

kernel void celeg_attention_merge_semantics_probe(
    device float* output [[buffer(0)]],
    constant float& destination_maximum [[buffer(1)]],
    constant float& destination_denominator [[buffer(2)]],
    constant float& source_maximum [[buffer(3)]],
    constant float& source_denominator [[buffer(4)]],
    constant float& destination_accumulator [[buffer(5)]],
    constant float& source_accumulator [[buffer(6)]],
    constant float& global_maximum [[buffer(7)]]) {
    const CelegMergeTransition transition = celeg_merge_pair(
        destination_maximum, destination_denominator,
        source_maximum, source_denominator);
    output[0] = transition.maximum;
    output[1] = transition.denominator;
    output[2] = transition.destination_scale;
    output[3] = transition.source_scale;
    output[4] = destination_accumulator * transition.destination_scale +
        source_accumulator * transition.source_scale;
    output[5] = celeg_partial_rescale(source_maximum, global_maximum);
}

struct CelegOnlineTransition {
    float maximum;
    float denominator;
    float previous_scale;
    float current_scale;
};

CelegOnlineTransition celeg_online_transition(
    float previous_maximum, float previous_denominator, float score) {
    const float maximum = max(previous_maximum, score);
    const float previous_scale = isfinite(previous_maximum)
        ? exp(previous_maximum - maximum) : 0.0f;
    const float current_scale = exp(score - maximum);
    return CelegOnlineTransition{
        maximum,
        previous_denominator * previous_scale + current_scale,
        previous_scale,
        current_scale};
}

kernel void celeg_attention_online_semantics_probe(
    device float* output [[buffer(0)]],
    constant float& previous_maximum [[buffer(1)]],
    constant float& previous_denominator [[buffer(2)]],
    constant float& score [[buffer(3)]],
    constant float& accumulator [[buffer(4)]],
    constant float& value [[buffer(5)]]) {
    const CelegOnlineTransition transition = celeg_online_transition(
        previous_maximum, previous_denominator, score);
    output[0] = transition.maximum;
    output[1] = transition.denominator;
    output[2] = transition.previous_scale;
    output[3] = transition.current_scale;
    output[4] = accumulator * transition.previous_scale +
        value * transition.current_scale;
}

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
        const uint safe_exact = max(max_exact, 1u);
        const uint safe_distance = max(distance, max_exact);
        const uint safe_max_distance = max(max_distance, max_exact + 1);
        const float denominator = log(
            static_cast<float>(safe_max_distance) /
            static_cast<float>(safe_exact));
        const float logarithmic = denominator == 0.0f ? 0.0f : log(
            static_cast<float>(safe_distance) /
            static_cast<float>(safe_exact)) / denominator;
        bucket = max_exact + static_cast<uint>(
            logarithmic * static_cast<float>(directional_buckets - max_exact));
        bucket = min(bucket, directional_buckets - 1);
    }
    if (positive) bucket += directional_buckets;
    return bucket;
}

kernel void celeg_attention_bias_semantics_probe(
    device uint* bucket_out [[buffer(0)]],
    device float* alibi_out [[buffer(1)]],
    constant int& query_position [[buffer(2)]],
    constant int& key_position [[buffer(3)]],
    constant uint& bucket_count [[buffer(4)]],
    constant uint& max_distance [[buffer(5)]],
    constant uint& bidirectional [[buffer(6)]],
    constant float* slopes [[buffer(7)]]) {
    bucket_out[0] = celeg_relative_position_bucket(
        query_position, key_position, bucket_count, max_distance, bidirectional);
    const CelegAttentionAlibiBias alibi{slopes};
    alibi_out[0] = alibi.value(
        0, static_cast<uint>(query_position), static_cast<uint>(key_position));
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
