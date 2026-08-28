#include <metal_stdlib>

using namespace metal;

kernel void celeg_embedding(device const float* table [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& token [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = table[static_cast<size_t>(token) * width + index];
}

kernel void celeg_matvec(device const float* weights [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = 0; col < cols; ++col) sum += weights[base + col] * input[col];
    output[row] = sum;
}

float celeg_half_to_float(ushort bits) {
    const uint sign = static_cast<uint>(bits & 0x8000u) << 16;
    const uint exponent = (static_cast<uint>(bits) >> 10) & 0x1fu;
    const uint fraction = static_cast<uint>(bits) & 0x3ffu;
    if (exponent == 0) {
        if (fraction == 0) return as_type<float>(sign);
        uint normalized = fraction;
        uint shift = 0;
        while ((normalized & 0x400u) == 0) {
            normalized <<= 1;
            ++shift;
        }
        normalized &= 0x3ffu;
        return as_type<float>(sign | ((127u - 14u - shift) << 23) |
                              (normalized << 13));
    }
    if (exponent == 0x1fu) {
        return as_type<float>(sign | 0x7f800000u | (fraction << 13));
    }
    return as_type<float>(sign | ((exponent + 112u) << 23) | (fraction << 13));
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

kernel void celeg_matvec_q4k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint block_index = 0; block_index < cols / 256; ++block_index) {
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        for (uint sub = 0; sub < 8; ++sub) {
            uchar scale = 0;
            uchar minimum = 0;
            celeg_q4k_scale_min(block + 4, sub, scale, minimum);
            const uint base = block_index * 256 + sub * 32;
            for (uint index = 0; index < 32; ++index) {
                const float value = d * static_cast<float>(scale) *
                                    static_cast<float>(celeg_q4k_value(block, sub * 32 + index)) -
                                    dmin * static_cast<float>(minimum);
                sum += value * input[base + index];
            }
        }
    }
    output[row] = sum;
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

kernel void celeg_matvec_q6k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint block_index = 0; block_index < cols / 256; ++block_index) {
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 210;
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        const uint base = block_index * 256;
        for (uint sub = 0; sub < 16; ++sub) {
            const float scale = d * static_cast<float>(static_cast<char>(block[192 + sub]));
            for (uint index = 0; index < 16; ++index) {
                const uint column = sub * 16 + index;
                const float value = scale * static_cast<float>(
                    celeg_q6k_value(block, column) - 32);
                sum += value * input[base + column];
            }
        }
    }
    output[row] = sum;
}

kernel void celeg_embedding_q4k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 144 + block_index * 144;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    const uint sub = column >> 5;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q4k_scale_min(block + 4, sub, scale, minimum);
    output[index] = d * static_cast<float>(scale) *
                    static_cast<float>(celeg_q4k_value(block, column)) -
                    dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q6k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 210 + block_index * 210;
    const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                        (static_cast<ushort>(block[209]) << 8));
    const uint sub = column / 16;
    const float scale = d * static_cast<float>(static_cast<char>(block[192 + sub]));
    output[index] = scale * static_cast<float>(celeg_q6k_value(block, column) - 32);
}

kernel void celeg_rmsnorm(device const float* input [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& width [[buffer(3)]],
                          constant float& epsilon [[buffer(4)]],
                          uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < width; ++i) sum += input[i] * input[i];
    const float inverse = rsqrt(sum / static_cast<float>(width) + epsilon);
    for (uint i = 0; i < width; ++i) output[i] = input[i] * inverse * weight[i];
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

kernel void celeg_qk_norm_rope(device float* query [[buffer(0)]],
                               device const float* query_weight [[buffer(1)]],
                               device float* key [[buffer(2)]],
                               device const float* key_weight [[buffer(3)]],
                               constant uint& query_heads [[buffer(4)]],
                               constant uint& key_heads [[buffer(5)]],
                               constant uint& head_dim [[buffer(6)]],
                               constant uint& position [[buffer(7)]],
                               constant float& theta [[buffer(8)]],
                               constant float& query_scale [[buffer(9)]],
                               constant float& query_epsilon [[buffer(10)]],
                               constant float& key_epsilon [[buffer(11)]],
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
    }
}

kernel void celeg_store_kv(device const float* key [[buffer(0)]],
                           device const float* value [[buffer(1)]],
                           device float* key_cache [[buffer(2)]],
                           device float* value_cache [[buffer(3)]],
                           constant uint& position [[buffer(4)]],
                           constant uint& width [[buffer(5)]],
                           uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const size_t offset = static_cast<size_t>(position) * width + index;
    key_cache[offset] = key[index];
    value_cache[offset] = value[index];
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
        const size_t key_base = static_cast<size_t>(position) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = static_cast<size_t>(position) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        denominator += exp(score * scale - maximum);
    }
    float value = 0.0f;
    for (uint position = 0; position < sequence_length; ++position) {
        const size_t key_base = static_cast<size_t>(position) * key_width +
                                static_cast<size_t>(key_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) score += query[query_base + d] * key_cache[key_base + d];
        value += exp(score * scale - maximum) * value_cache[key_base + dimension];
    }
    output[index] = value / denominator;
}
