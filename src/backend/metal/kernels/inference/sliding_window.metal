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
    uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    if (index >= width) return;
    const uint head = index / head_dim;
    const uint dimension = index % head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    const uint start = sequence_length > window_size
        ? sequence_length - window_size : 0;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        denominator += exp(score * scale - maximum);
    }
    float value = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        value += exp(score * scale - maximum) *
            value_cache[key_base + dimension];
    }
    output[index] = value / denominator;
}

kernel void celeg_attention_sliding_cooperative(
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
    uint lane [[thread_index_in_simdgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    if (head >= query_heads || sequence_length > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    const uint start = sequence_length > window_size
        ? sequence_length - window_size : 0;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) {
            partial += query[query_base + d] * key_cache[key_base + d];
        }
        const float score = simd_sum(partial) * scale;
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
            const size_t key_base =
                (static_cast<size_t>(position / page_tokens) * page_tokens +
                 position % page_tokens) * key_width +
                static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator *
                value_cache[key_base + dimension];
        }
        output[query_base + dimension] = value;
    }
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
    uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    const uint row = index / width;
    const uint dimension = index % width;
    if (row >= rows) return;
    const uint head = dimension / head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const uint sequence_length = base_position + row + 1;
    const uint start = sequence_length > window_size
        ? sequence_length - window_size : 0;
    const size_t query_base = static_cast<size_t>(row) * width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        denominator += exp(score * scale - maximum);
    }
    float value_sum = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_base + d] * key_cache[key_base + d];
        }
        const float probability = exp(score * scale - maximum) / denominator;
        value_sum += probability *
            value_cache[key_base + (dimension % head_dim)];
    }
    output[static_cast<size_t>(row) * width + dimension] = value_sum;
}

kernel void celeg_attention_batch_sliding_cooperative(
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
    uint lane [[thread_index_in_simdgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.y;
    const uint head = grid.x;
    if (row >= rows || head >= query_heads ||
        base_position + row + 1 > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim;
    const size_t query_base = static_cast<size_t>(row) * query_width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    const uint sequence_length = base_position + row + 1;
    const uint start = sequence_length > window_size
        ? sequence_length - window_size : 0;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base =
            (static_cast<size_t>(position / page_tokens) * page_tokens +
             position % page_tokens) * key_width +
            static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) {
            partial += query[query_base + d] * key_cache[key_base + d];
        }
        const float score = simd_sum(partial) * scale;
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
            const size_t key_base =
                (static_cast<size_t>(position / page_tokens) * page_tokens +
                 position % page_tokens) * key_width +
                static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator *
                value_cache[key_base + dimension];
        }
        output[static_cast<size_t>(row) * query_width +
               static_cast<size_t>(head) * head_dim + dimension] = value;
    }
}
