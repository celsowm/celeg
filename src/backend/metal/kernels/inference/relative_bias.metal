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
    uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    if (index >= width) return;
    const uint head = index / head_dim;
    const uint dimension = index % head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const uint query_position = sequence_length - 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        maximum = max(maximum, biased);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        denominator += exp(biased - maximum);
    }
    float value = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        value += exp(biased - maximum) * value_cache[key_base + dimension];
    }
    output[index] = value / denominator;
}

kernel void celeg_attention_relative_bias_cooperative(
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
    uint lane [[thread_index_in_simdgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    if (head >= query_heads || sequence_length > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const uint query_position = sequence_length - 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) {
            partial += query[query_base + d] * key_cache[key_base + d];
        }
        const float bias = celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        const float score = simd_sum(partial) * scale + bias;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) {
            maximum_value = max(maximum_value, scores[position]);
        }
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            denominator_value += exp(scores[position] - maximum_value);
        }
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
                position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator *
                value_cache[key_base + dimension];
        }
        output[query_base + dimension] = value;
    }
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
    uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    const uint row = index / width;
    const uint dimension = index % width;
    if (row >= rows) return;
    const uint head = dimension / head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const uint query_position = base_position + row;
    const uint sequence_length = query_position + 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    const size_t query_base = static_cast<size_t>(row) * width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        maximum = max(maximum, biased);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        denominator += exp(biased - maximum);
    }
    float value_sum = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float biased = score * scale + celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        const float probability = exp(biased - maximum) / denominator;
        value_sum += probability * value_cache[key_base + (dimension % head_dim)];
    }
    output[static_cast<size_t>(row) * width + dimension] = value_sum;
}

kernel void celeg_attention_batch_relative_bias_cooperative(
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
    uint lane [[thread_index_in_simdgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.y;
    const uint head = grid.x;
    if (row >= rows || head >= query_heads || base_position + row + 1 > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const uint query_position = base_position + row;
    const uint sequence_length = query_position + 1;
    const uint start = window_size > 0 && sequence_length > window_size
        ? sequence_length - window_size : 0;
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim;
    const size_t query_base = static_cast<size_t>(row) * query_width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) {
            partial += query[query_base + d] * key_cache[key_base + d];
        }
        const float bias = celeg_relative_position_bias(
            bias_values, head, static_cast<int>(query_position), static_cast<int>(position),
            bucket_count, max_distance, bidirectional);
        const float score = simd_sum(partial) * scale + bias;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) {
            maximum_value = max(maximum_value, scores[position]);
        }
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            denominator_value += exp(scores[position] - maximum_value);
        }
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
                position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator *
                value_cache[key_base + dimension];
        }
        output[static_cast<size_t>(row) * query_width +
               static_cast<size_t>(head) * head_dim + dimension] = value;
    }
}
