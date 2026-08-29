kernel void celeg_embedding_batch(device const float* table [[buffer(0)]],
                                  device float* output [[buffer(1)]],
                                  constant uint& width [[buffer(2)]],
                                  device const uint* tokens [[buffer(3)]],
                                  uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    output[static_cast<size_t>(token) * width + column] =
        table[static_cast<size_t>(tokens[token]) * width + column];
}

kernel void celeg_embedding_f16_batch(device const half* table [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& width [[buffer(2)]],
                                      device const uint* tokens [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    output[static_cast<size_t>(token) * width + column] = static_cast<float>(
        table[static_cast<size_t>(tokens[token]) * width + column]);
}

kernel void celeg_embedding_bf16_batch(device const ushort* table [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& width [[buffer(2)]],
                                       device const uint* tokens [[buffer(3)]],
                                       uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    output[static_cast<size_t>(token) * width + column] = celeg_bf16_to_float(
        table[static_cast<size_t>(tokens[token]) * width + column]);
}

kernel void celeg_embedding_q4_0_batch(device const uchar* weights [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& width [[buffer(2)]],
                                       device const uint* tokens [[buffer(3)]],
                                       uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    const device uchar* block = weights + static_cast<size_t>(tokens[token]) *
        (width / 32) * 18 + static_cast<size_t>(column / 32) * 18;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[static_cast<size_t>(token) * width + column] = d *
        celeg_q4_0_value(block, column);
}

kernel void celeg_embedding_q5k_batch(device const uchar* weights [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& width [[buffer(2)]],
                                      device const uint* tokens [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    const uint block_index = column / 256;
    const uint within = column & 255;
    const device uchar* block = weights + static_cast<size_t>(tokens[token]) *
        (width / 256) * 176 + static_cast<size_t>(block_index) * 176;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q5k_scale_min(block + 4, within >> 5, scale, minimum);
    output[static_cast<size_t>(token) * width + column] =
        d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
        dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q8_0_batch(device const uchar* weights [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& width [[buffer(2)]],
                                       device const uint* tokens [[buffer(3)]],
                                       uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    const device uchar* block = weights + static_cast<size_t>(tokens[token]) *
        (width / 32) * 34 + static_cast<size_t>(column / 32) * 34;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[static_cast<size_t>(token) * width + column] = d *
        static_cast<float>(static_cast<char>(block[2 + (column & 31)]));
}

kernel void celeg_embedding_q4k_batch(device const uchar* weights [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& width [[buffer(2)]],
                                      device const uint* tokens [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    const uint block_index = column / 256;
    const uint within = column & 255;
    const device uchar* block = weights + static_cast<size_t>(tokens[token]) *
        (width / 256) * 144 + static_cast<size_t>(block_index) * 144;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q4k_scale_min(block + 4, within >> 5, scale, minimum);
    output[static_cast<size_t>(token) * width + column] =
        d * static_cast<float>(scale) * celeg_q4k_value(block, within) -
        dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q6k_batch(device const uchar* weights [[buffer(0)]],
                                      device float* output [[buffer(1)]],
                                      constant uint& width [[buffer(2)]],
                                      device const uint* tokens [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
    const uint token = index / width;
    const uint column = index % width;
    const uint block_index = column / 256;
    const uint within = column & 255;
    const device uchar* block = weights + static_cast<size_t>(tokens[token]) *
        (width / 256) * 210 + static_cast<size_t>(block_index) * 210;
    const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                        (static_cast<ushort>(block[209]) << 8));
    const float scale = d * static_cast<float>(static_cast<char>(block[192 + within / 16]));
    output[static_cast<size_t>(token) * width + column] = scale *
        static_cast<float>(celeg_q6k_value(block, within) - 32);
}

kernel void celeg_rmsnorm_batch(device const float* input [[buffer(0)]],
                                device const float* weight [[buffer(1)]],
                                device float* output [[buffer(2)]],
                                constant uint& rows [[buffer(3)]],
                                constant uint& width [[buffer(4)]],
                                constant float& epsilon [[buffer(5)]],
                                uint index [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint token [[threadgroup_position_in_grid]]) {
    if (token >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(token) * width;
    for (uint i = index; i < width; i += 256) sum += input[base + i] * input[base + i];
    threadgroup float partial[8];
    threadgroup float inverse;
    const float reduced = simd_sum(sum);
    if (lane == 0) partial[simd] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index == 0) {
        float total = 0.0f;
        for (uint group = 0; group < 8; ++group) total += partial[group];
        inverse = rsqrt(total / static_cast<float>(width) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = index; i < width; i += 256) {
        output[base + i] = input[base + i] * inverse * weight[i];
    }
}

kernel void celeg_copy_batch(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = input[index];
}

kernel void celeg_residual_batch(device const float* input [[buffer(0)]],
                                 device const float* residual [[buffer(1)]],
                                 device float* output [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 constant float& multiplier [[buffer(4)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = input[index] * multiplier + residual[index];
}

kernel void celeg_swiglu_batch(device const float* gate_up [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& width [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint token = index / width;
    const uint column = index % width;
    const size_t base = static_cast<size_t>(token) * width * 2;
    const float gate = gate_up[base + column];
    const float up = gate_up[base + width + column];
    output[static_cast<size_t>(token) * width + column] =
        gate / (1.0f + exp(-gate)) * up;
}

kernel void celeg_shortconv_batch(device const float* projected [[buffer(0)]],
                                  device const float* taps [[buffer(1)]],
                                  device float* state [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& width [[buffer(5)]],
                                  constant uint& cache_length [[buffer(6)]],
                                  constant uint& base_position [[buffer(7)]],
                                  uint channel [[thread_position_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    const uint actual_channel = group * 256 + channel;
    if (actual_channel >= width) return;
    for (uint row = 0; row < rows; ++row) {
        const uint position = base_position + row;
        const uint cursor = position % cache_length;
        const size_t projected_base = static_cast<size_t>(row) * 3 * width;
        const size_t state_index = static_cast<size_t>(cursor) * width + actual_channel;
        state[state_index] = projected[projected_base + actual_channel] *
            projected[projected_base + 2 * width + actual_channel];
        float convolution = 0.0f;
        for (uint tap = 0; tap < cache_length; ++tap) {
            const uint slot = (cursor + 1 + tap) % cache_length;
            convolution += state[static_cast<size_t>(slot) * width + actual_channel] *
                taps[static_cast<size_t>(tap) * width + actual_channel];
        }
        output[static_cast<size_t>(row) * width + actual_channel] =
            projected[projected_base + width + actual_channel] * convolution;
    }
}

kernel void celeg_qk_norm_rope_batch(device float* query [[buffer(0)]],
                                    device const float* query_weight [[buffer(1)]],
                                    device float* key [[buffer(2)]],
                                    device const float* key_weight [[buffer(3)]],
                                    constant uint& rows [[buffer(4)]],
                                    constant uint& query_heads [[buffer(5)]],
                                    constant uint& key_heads [[buffer(6)]],
                                    constant uint& head_dim [[buffer(7)]],
                                    constant uint& base_position [[buffer(8)]],
                                    constant float& theta [[buffer(9)]],
                                    constant float& query_scale [[buffer(10)]],
                                    constant float& query_epsilon [[buffer(11)]],
                                    constant float& key_epsilon [[buffer(12)]],
                                    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(token) * query_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= inverse * query_weight[d];
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(base_position + token) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t offset = base + 2 * pair;
            const float x = query[offset];
            const float y = query[offset + 1];
            query[offset] = (x * c - y * s) * query_scale;
            query[offset + 1] = (x * s + y * c) * query_scale;
        }
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(token) * key_heads * head_dim +
            static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += key[base + d] * key[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint d = 0; d < head_dim; ++d) key[base + d] *= inverse * key_weight[d];
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(base_position + token) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t offset = base + 2 * pair;
            const float x = key[offset];
            const float y = key[offset + 1];
            key[offset] = x * c - y * s;
            key[offset + 1] = x * s + y * c;
        }
    }
}

kernel void celeg_store_kv_batch(device const float* key [[buffer(0)]],
                                 device const float* value [[buffer(1)]],
                                 device float* key_cache [[buffer(2)]],
                                 device float* value_cache [[buffer(3)]],
                                 constant uint& rows [[buffer(4)]],
                                 constant uint& base_position [[buffer(5)]],
                                 constant uint& width [[buffer(6)]],
                                 constant uint& page_tokens [[buffer(7)]],
                                 uint index [[thread_position_in_grid]]) {
    const uint row = index / width;
    const uint column = index % width;
    if (row >= rows) return;
    const uint position = base_position + row;
    const size_t offset = (static_cast<size_t>(position / page_tokens) * page_tokens +
        position % page_tokens) * width + column;
    key_cache[offset] = key[static_cast<size_t>(row) * width + column];
    value_cache[offset] = value[static_cast<size_t>(row) * width + column];
}

kernel void celeg_attention_batch(device const float* query [[buffer(0)]],
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
                                  uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    const uint row = index / width;
    const uint dimension = index % width;
    if (row >= rows) return;
    const uint head = dimension / head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const uint sequence_length = base_position + row + 1;
    const size_t query_base = static_cast<size_t>(row) * width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] *
            key_cache[key_base + d];
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] *
            key_cache[key_base + d];
        denominator += exp(score * scale - maximum);
    }
    float value_sum = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] *
            key_cache[key_base + d];
        const float probability = exp(score * scale - maximum) / denominator;
        value_sum += probability *
            value_cache[key_base + (dimension % head_dim)];
    }
    output[static_cast<size_t>(row) * width + dimension] = value_sum;
}

kernel void celeg_attention_batch_cooperative(device const float* query [[buffer(0)]],
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
                                              uint lane [[thread_index_in_simdgroup]],
                                              uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.y;
    const uint head = grid.x;
    if (row >= rows || head >= query_heads || base_position + row + 1 > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_width = static_cast<size_t>(query_heads) * head_dim;
    const size_t query_base = static_cast<size_t>(row) * query_width +
        static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    const uint sequence_length = base_position + row + 1;
    threadgroup float scores[1024];
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
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
        for (uint position = 0; position < sequence_length; ++position) {
            maximum_value = max(maximum_value, scores[position]);
        }
        maximum = maximum_value;
        float denominator_value = 0.0f;
        for (uint position = 0; position < sequence_length; ++position) {
            denominator_value += exp(scores[position] - maximum_value);
        }
        denominator = denominator_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = 0; position < sequence_length; ++position) {
            const size_t key_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
                position % page_tokens) * key_width + static_cast<size_t>(key_head) * head_dim;
            value += exp(scores[position] - maximum) / denominator *
                value_cache[key_base + dimension];
        }
        output[static_cast<size_t>(row) * query_width + static_cast<size_t>(head) * head_dim +
               dimension] = value;
    }
}

