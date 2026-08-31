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
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        denominator += exp(score * scale - maximum);
    }
    float value = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        value += exp(score * scale - maximum) * value_cache[key_base + dimension];
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
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) partial += query[query_base + d] * key_cache[key_base + d];
        const float score = simd_sum(partial) * scale;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) maximum_value = max(maximum_value, scores[position]);
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) denominator_value += exp(scores[position] - maximum_value);
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
                position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator * value_cache[key_base + dimension];
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
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    const size_t query_base = static_cast<size_t>(row) * width + static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        denominator += exp(score * scale - maximum);
    }
    float value_sum = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float probability = exp(score * scale - maximum) / denominator;
        value_sum += probability * value_cache[key_base + (dimension % head_dim)];
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
    if (row >= rows || head >= query_heads || base_position + row + 1 > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim;
    const size_t query_base = static_cast<size_t>(row) * query_width + static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    const uint sequence_length = base_position + row + 1;
    const uint start = sequence_length > window_size ? sequence_length - window_size : 0;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) partial += query[query_base + d] * key_cache[key_base + d];
        const float score = simd_sum(partial) * scale;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) maximum_value = max(maximum_value, scores[position]);
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) denominator_value += exp(scores[position] - maximum_value);
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator * value_cache[key_base + dimension];
        }
        output[static_cast<size_t>(row) * query_width + static_cast<size_t>(head) * head_dim + dimension] = value;
    }
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
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        maximum = max(maximum, biased);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        denominator += exp(biased - maximum);
    }
    float value = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        value += exp(biased - maximum) * value_cache[key_base + dimension];
    }
    output[index] = value / denominator;
}

kernel void celeg_attention_alibi_cooperative(
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
        for (uint d = lane; d < head_dim; d += 32) partial += query[query_base + d] * key_cache[key_base + d];
        const float bias = -slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        const float score = simd_sum(partial) * scale + bias;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) maximum_value = max(maximum_value, scores[position]);
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) denominator_value += exp(scores[position] - maximum_value);
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
                position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator * value_cache[key_base + dimension];
        }
        output[query_base + dimension] = value;
    }
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
    const size_t query_base = static_cast<size_t>(row) * width + static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        maximum = max(maximum, biased);
    }
    float denominator = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        denominator += exp(biased - maximum);
    }
    float value_sum = 0.0f;
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        const float biased = score * scale - slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        const float probability = exp(biased - maximum) / denominator;
        value_sum += probability * value_cache[key_base + (dimension % head_dim)];
    }
    output[static_cast<size_t>(row) * width + dimension] = value_sum;
}

kernel void celeg_attention_batch_alibi_cooperative(
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
    const size_t query_base = static_cast<size_t>(row) * query_width + static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    threadgroup float scores[1024];
    for (uint position = start; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float partial = 0.0f;
        for (uint d = lane; d < head_dim; d += 32) partial += query[query_base + d] * key_cache[key_base + d];
        const float bias = -slopes[head] *
            static_cast<float>(abs(static_cast<int>(query_position) - static_cast<int>(position)));
        const float score = simd_sum(partial) * scale + bias;
        if (lane == 0) scores[position] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float maximum;
    threadgroup float denominator;
    if (lane == 0) {
        float maximum_value = -INFINITY;
        for (uint position = start; position < sequence_length; ++position) maximum_value = max(maximum_value, scores[position]);
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) denominator_value += exp(scores[position] - maximum_value);
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = start; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens + position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator * value_cache[key_base + dimension];
        }
        output[static_cast<size_t>(row) * query_width + static_cast<size_t>(head) * head_dim + dimension] = value;
    }
}
