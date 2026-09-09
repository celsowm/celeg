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

float celeg_alibi_bias(float slope, int query_position, int key_position) {
    return -slope * static_cast<float>(abs(query_position - key_position));
}

kernel void celeg_attention_bias_semantics_probe(
    device uint* bucket_out [[buffer(0)]],
    device float* alibi_out [[buffer(1)]],
    constant int& query_position [[buffer(2)]],
    constant int& key_position [[buffer(3)]],
    constant uint& bucket_count [[buffer(4)]],
    constant uint& max_distance [[buffer(5)]],
    constant uint& bidirectional [[buffer(6)]],
    constant float& slope [[buffer(7)]]) {
    bucket_out[0] = celeg_relative_position_bucket(
        query_position, key_position, bucket_count, max_distance, bidirectional);
    alibi_out[0] = celeg_alibi_bias(slope, query_position, key_position);
}
