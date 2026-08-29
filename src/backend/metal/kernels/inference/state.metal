kernel void celeg_rmsnorm(device const float* input [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& width [[buffer(3)]],
                          constant float& epsilon [[buffer(4)]],
                          uint index [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = index; i < width; i += 256) sum += input[i] * input[i];
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
    for (uint i = index; i < width; i += 256) output[i] = input[i] * inverse * weight[i];
}

kernel void celeg_rmsnorm_save(device const float* input [[buffer(0)]],
                               device float* residual [[buffer(1)]],
                               device const float* weight [[buffer(2)]],
                               device float* output [[buffer(3)]],
                               constant uint& width [[buffer(4)]],
                               constant float& epsilon [[buffer(5)]],
                               uint index [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = index; i < width; i += 256) sum += input[i] * input[i];
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
        residual[i] = input[i];
        output[i] = input[i] * inverse * weight[i];
    }
}

kernel void celeg_residual_rmsnorm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    uint index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = index; i < width; i += 256) {
        const float value = input[i] * multiplier + residual[i];
        sum += value * value;
    }
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
        const float value = input[i] * multiplier + residual[i];
        output[i] = value;
        normed[i] = value * inverse * weight[i];
    }
}

kernel void celeg_copy(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = input[index];
}

kernel void celeg_residual(device const float* input [[buffer(0)]],
                           device const float* residual [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant uint& count [[buffer(3)]],
                           constant float& multiplier [[buffer(4)]],
                           uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = input[index] * multiplier + residual[index];
}

kernel void celeg_scale(device float* data [[buffer(0)]],
                        constant uint& count [[buffer(1)]],
                        constant float& multiplier [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
    if (index < count) data[index] *= multiplier;
}

kernel void celeg_weighted_add(device const float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               constant float& weight [[buffer(3)]],
                               uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] += input[index] * weight;
}

kernel void celeg_swiglu(device const float* gate_up [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& width [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const float gate = gate_up[index];
    const float up = gate_up[width + index];
    output[index] = gate / (1.0f + exp(-gate)) * up;
}

kernel void celeg_shortconv(device const float* projected [[buffer(0)]],
                            device const float* taps [[buffer(1)]],
                            device float* state [[buffer(2)]],
                            device float* output [[buffer(3)]],
                            constant uint& width [[buffer(4)]],
                            constant uint& cache_length [[buffer(5)]],
                            constant uint& position [[buffer(6)]],
                            uint channel [[thread_position_in_grid]]) {
    if (channel >= width) return;
    const uint cursor = position % cache_length;
    const float value = projected[2 * width + channel] * projected[channel];
    state[static_cast<size_t>(cursor) * width + channel] = value;
    float convolution = 0.0f;
    for (uint tap = 0; tap < cache_length; ++tap) {
        const uint slot = (cursor + 1 + tap) % cache_length;
        convolution += state[static_cast<size_t>(slot) * width + channel] *
                       taps[static_cast<size_t>(tap) * width + channel];
    }
    output[channel] = projected[width + channel] * convolution;
}

kernel void celeg_qk_norm_rope_store_kv(
    device float* query [[buffer(0)]],
    device const float* query_weight [[buffer(1)]],
    device float* key [[buffer(2)]],
    device const float* key_weight [[buffer(3)]],
    device const float* value [[buffer(4)]],
    device float* key_cache [[buffer(5)]],
    device float* value_cache [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& key_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& position [[buffer(10)]],
    constant float& theta [[buffer(11)]],
    constant float& query_scale [[buffer(12)]],
    constant float& query_epsilon [[buffer(13)]],
    constant float& key_epsilon [[buffer(14)]],
    constant uint& page_tokens [[buffer(15)]],
    uint head [[thread_position_in_grid]]) {
    if (head < query_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += query[base + d] * query[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + query_epsilon);
        for (uint d = 0; d < head_dim; ++d) query[base + d] *= inverse * query_weight[d];
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = cos(angle);
            const float sine = sin(angle);
            const size_t offset = base + 2 * pair;
            const float x = query[offset];
            const float y = query[offset + 1];
            query[offset] = (x * cosine - y * sine) * query_scale;
            query[offset + 1] = (x * sine + y * cosine) * query_scale;
        }
    }
    if (head < key_heads) {
        const size_t base = static_cast<size_t>(head) * head_dim;
        float sum = 0.0f;
        for (uint d = 0; d < head_dim; ++d) sum += key[base + d] * key[base + d];
        const float inverse = rsqrt(sum / static_cast<float>(head_dim) + key_epsilon);
        for (uint d = 0; d < head_dim; ++d) key[base + d] *= inverse * key_weight[d];
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float cosine = cos(angle);
            const float sine = sin(angle);
            const size_t offset = base + 2 * pair;
            const float x = key[offset];
            const float y = key[offset + 1];
            key[offset] = x * cosine - y * sine;
            key[offset + 1] = x * sine + y * cosine;
        }
        const size_t cache_base = (static_cast<size_t>(position / page_tokens) * page_tokens +
            position % page_tokens) * static_cast<size_t>(key_heads) * head_dim + base;
        for (uint d = 0; d < head_dim; ++d) {
            key_cache[cache_base + d] = key[base + d];
            value_cache[cache_base + d] = value[base + d];
        }
    }
}

kernel void celeg_attention(device const float* query [[buffer(0)]],
                            device const float* key_cache [[buffer(1)]],
                            device const float* value_cache [[buffer(2)]],
                            device float* output [[buffer(3)]],
                            constant uint& sequence_length [[buffer(4)]],
                            constant uint& query_heads [[buffer(5)]],
                            constant uint& key_heads [[buffer(6)]],
                            constant uint& head_dim [[buffer(7)]],
                            constant float& scale [[buffer(8)]],
                            constant uint& page_tokens [[buffer(9)]],
                            uint index [[thread_position_in_grid]]) {
    const uint width = query_heads * head_dim;
    if (index >= width) return;
    const uint head = index / head_dim;
    const uint dimension = index % head_dim;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
    float maximum = -INFINITY;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t page = position / page_tokens;
        const size_t slot = position % page_tokens;
        const size_t key_base = (page * page_tokens + slot) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t page = position / page_tokens;
        const size_t slot = position % page_tokens;
        const size_t key_base = (page * page_tokens + slot) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        denominator += exp(score * scale - maximum);
    }
    float value = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t page = position / page_tokens;
        const size_t slot = position % page_tokens;
        const size_t key_base = (page * page_tokens + slot) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        value += exp(score * scale - maximum) * value_cache[key_base + dimension];
    }
    output[index] = value / denominator;
}

kernel void celeg_attention_cooperative(device const float* query [[buffer(0)]],
                                        device const float* key_cache [[buffer(1)]],
                                        device const float* value_cache [[buffer(2)]],
                                        device float* output [[buffer(3)]],
                                        constant uint& sequence_length [[buffer(4)]],
                                        constant uint& query_heads [[buffer(5)]],
                                        constant uint& key_heads [[buffer(6)]],
                                        constant uint& head_dim [[buffer(7)]],
                                        constant float& scale [[buffer(8)]],
                                        constant uint& page_tokens [[buffer(9)]],
                                        uint lane [[thread_index_in_simdgroup]],
                                        uint2 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    if (head >= query_heads || sequence_length > 1024) return;
    const uint key_head = head / (query_heads / key_heads);
    const size_t query_base = static_cast<size_t>(head) * head_dim;
    const size_t key_width = static_cast<size_t>(key_heads) * head_dim;
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
        output[query_base + dimension] = value;
    }
}

float celeg_silu(float value) {
    return value / (1.0f + exp(-value));
}

float celeg_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return exp(value);
    return log(1.0f + exp(value));
}

kernel void celeg_gated_delta(
    device float* projected_qkv [[buffer(0)]],
    device const float* projected_z [[buffer(1)]],
    device const float* projected_b [[buffer(2)]],
    device const float* projected_a [[buffer(3)]],
    device const float* conv_weight [[buffer(4)]],
    device const float* dt_bias [[buffer(5)]],
    device const float* a_log [[buffer(6)]],
    device const float* norm_weight [[buffer(7)]],
    device float* conv_state [[buffer(8)]],
    device float* recurrent_state [[buffer(9)]],
    device float* output [[buffer(10)]],
    constant uint& conv_kernel [[buffer(11)]],
    constant uint& key_head_dim [[buffer(12)]],
    constant uint& value_head_dim [[buffer(13)]],
    constant uint& key_heads [[buffer(14)]],
    constant uint& value_heads [[buffer(15)]],
    constant float& epsilon [[buffer(16)]],
    constant uint& vector_decay [[buffer(17)]],
    constant uint& safe_decay [[buffer(18)]],
    constant float& decay_lower_bound [[buffer(19)]],
    constant uint& sigmoid_output_gate [[buffer(20)]],
    constant uint& a_log_needs_exp [[buffer(21)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    const uint key_width = key_heads * key_head_dim;
    const uint value_width = value_heads * value_head_dim;
    const uint conv_width = 2 * key_width + value_width;
    for (uint channel = 0; channel < conv_width; ++channel) {
        device float* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
        const device float* weight = conv_weight + static_cast<size_t>(channel) * conv_kernel;
        for (uint tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
        history[conv_kernel - 1] = projected_qkv[channel];
        float filtered = 0.0f;
        for (uint tap = 0; tap < conv_kernel; ++tap) filtered += history[tap] * weight[tap];
        projected_qkv[channel] = celeg_silu(filtered);
    }
    const uint repeat = value_heads / key_heads;
    for (uint value_head = 0; value_head < value_heads; ++value_head) {
        const uint key_head = value_head / repeat;
        const device float* q = projected_qkv + static_cast<size_t>(key_head) * key_head_dim;
        const device float* k = projected_qkv + key_width +
                                static_cast<size_t>(key_head) * key_head_dim;
        const device float* v = projected_qkv + 2 * key_width +
                                static_cast<size_t>(value_head) * value_head_dim;
        device float* state = recurrent_state +
            static_cast<size_t>(value_head) * key_head_dim * value_head_dim;
        const float beta = 1.0f / (1.0f + exp(-projected_b[value_head]));
        float key_norm = 0.0f;
        float query_norm = 0.0f;
        for (uint key_dimension = 0; key_dimension < key_head_dim; ++key_dimension) {
            key_norm += k[key_dimension] * k[key_dimension];
            query_norm += q[key_dimension] * q[key_dimension];
        }
        key_norm = sqrt(key_norm + epsilon);
        query_norm = sqrt(query_norm + epsilon);
        for (uint key_dimension = 0; key_dimension < key_head_dim; ++key_dimension) {
            const uint decay_index = vector_decay
                ? key_head * key_head_dim + key_dimension : value_head;
            const float decay_base = a_log_needs_exp
                ? exp(a_log[value_head]) : a_log[value_head];
            const float dt = projected_a[decay_index] + dt_bias[decay_index];
            const float decay = safe_decay
                ? exp((1.0f / (1.0f + exp(-(decay_base * dt)))) * decay_lower_bound)
                : exp((a_log_needs_exp ? -decay_base : decay_base) * celeg_softplus(dt));
            for (uint value_dimension = 0; value_dimension < value_head_dim;
                 ++value_dimension) {
                const size_t state_index = static_cast<size_t>(key_dimension) *
                    value_head_dim + value_dimension;
                state[state_index] *= decay;
            }
        }
        for (uint value_dimension = 0; value_dimension < value_head_dim;
             ++value_dimension) {
            float memory = 0.0f;
            for (uint key_dimension = 0; key_dimension < key_head_dim; ++key_dimension) {
                memory += state[static_cast<size_t>(key_dimension) * value_head_dim +
                                value_dimension] *
                    (k[key_dimension] / key_norm);
            }
            output[static_cast<size_t>(value_head) * value_head_dim + value_dimension] =
                (v[value_dimension] - memory) * beta;
        }
        for (uint value_dimension = 0; value_dimension < value_head_dim;
             ++value_dimension) {
            const float delta = output[static_cast<size_t>(value_head) * value_head_dim +
                                       value_dimension];
            for (uint key_dimension = 0; key_dimension < key_head_dim; ++key_dimension) {
                const float normalized_key = k[key_dimension] / key_norm;
                state[static_cast<size_t>(key_dimension) * value_head_dim + value_dimension] +=
                    normalized_key * delta;
            }
        }
        for (uint value_dimension = 0; value_dimension < value_head_dim;
             ++value_dimension) {
            float value = 0.0f;
            for (uint key_dimension = 0; key_dimension < key_head_dim; ++key_dimension) {
                value += state[static_cast<size_t>(key_dimension) * value_head_dim +
                               value_dimension] * q[key_dimension] / query_norm;
            }
            output[static_cast<size_t>(value_head) * value_head_dim + value_dimension] =
                value / sqrt(static_cast<float>(key_head_dim));
        }
        float sum = 0.0f;
        const size_t output_base = static_cast<size_t>(value_head) * value_head_dim;
        for (uint value_dimension = 0; value_dimension < value_head_dim; ++value_dimension) {
            const float value = output[output_base + value_dimension];
            sum += value * value;
        }
        const float inverse = rsqrt(sum / static_cast<float>(value_head_dim) + epsilon);
        for (uint value_dimension = 0; value_dimension < value_head_dim; ++value_dimension) {
            const float gate = projected_z[output_base + value_dimension];
            const float gated = sigmoid_output_gate
                ? 1.0f / (1.0f + exp(-gate)) : celeg_silu(gate);
            output[output_base + value_dimension] =
                output[output_base + value_dimension] * inverse *
                norm_weight[value_dimension] * gated;
        }
    }
}

kernel void celeg_mamba2(
    device float* projected [[buffer(0)]],
    device const float* conv_weight [[buffer(1)]],
    device const float* conv_bias [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* a_log [[buffer(4)]],
    device const float* d [[buffer(5)]],
    device const float* norm_weight [[buffer(6)]],
    device float* conv_state [[buffer(7)]],
    device float* ssm_state [[buffer(8)]],
    device float* output [[buffer(9)]],
    constant uint& inner [[buffer(10)]],
    constant uint& state_size [[buffer(11)]],
    constant uint& num_heads [[buffer(12)]],
    constant uint& head_dim [[buffer(13)]],
    constant uint& group_count [[buffer(14)]],
    constant uint& conv_kernel [[buffer(15)]],
    constant float& epsilon [[buffer(16)]],
    constant uint& a_log_needs_exp [[buffer(17)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    const uint conv_width = inner + 2 * group_count * state_size;
    device float* xbc = projected + inner;
    for (uint channel = 0; channel < conv_width; ++channel) {
        device float* history = conv_state + static_cast<size_t>(channel) * conv_kernel;
        const device float* weight = conv_weight + static_cast<size_t>(channel) * conv_kernel;
        for (uint tap = 1; tap < conv_kernel; ++tap) history[tap - 1] = history[tap];
        history[conv_kernel - 1] = xbc[channel];
        float filtered = conv_bias[channel];
        for (uint tap = 0; tap < conv_kernel; ++tap) filtered += history[tap] * weight[tap];
        xbc[channel] = celeg_silu(filtered);
    }
    const uint group_size = num_heads / group_count;
    const uint dt_offset = inner + conv_width;
    for (uint head = 0; head < num_heads; ++head) {
        const float dt = celeg_softplus(projected[dt_offset + head] + dt_bias[head]);
        const float a = a_log_needs_exp ? -exp(a_log[head]) : a_log[head];
        const float decay = exp(dt * a);
        const uint group = head / group_size;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            const uint channel = head * head_dim + dimension;
            const float x = xbc[channel];
            const device float* b = xbc + inner + group * state_size;
            const device float* c = b + group_count * state_size;
            device float* state = ssm_state +
                (static_cast<size_t>(channel) * state_size);
            float value = 0.0f;
            for (uint state_dimension = 0; state_dimension < state_size;
                 ++state_dimension) {
                state[state_dimension] = decay * state[state_dimension] +
                    dt * b[state_dimension] * x;
                value += state[state_dimension] * c[state_dimension];
            }
            output[channel] = value + d[head] * x;
        }
    }
    for (uint dimension = 0; dimension < inner; ++dimension) {
        output[dimension] *= celeg_silu(projected[dimension]);
    }
    const uint norm_width = inner / group_count;
    for (uint group = 0; group < group_count; ++group) {
        const size_t base = static_cast<size_t>(group) * norm_width;
        float sum = 0.0f;
        for (uint dimension = 0; dimension < norm_width; ++dimension) {
            const float value = output[base + dimension];
            sum += value * value;
        }
        const float inverse = rsqrt(sum / static_cast<float>(norm_width) + epsilon);
        for (uint dimension = 0; dimension < norm_width; ++dimension) {
            output[base + dimension] *= inverse * norm_weight[base + dimension];
        }
    }
}
