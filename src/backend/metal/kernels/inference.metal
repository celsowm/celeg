#include <metal_stdlib>

using namespace metal;

float celeg_half_to_float(ushort bits);

kernel void celeg_embedding(device const float* table [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& width [[buffer(2)]],
                            constant uint& token [[buffer(3)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = table[static_cast<size_t>(token) * width + index];
}

kernel void celeg_embedding_f16(device const half* table [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = static_cast<float>(
        table[static_cast<size_t>(token) * width + index]);
}

float celeg_bf16_to_float(ushort bits) {
    return as_type<float>(static_cast<uint>(bits) << 16);
}

kernel void celeg_embedding_bf16(device const ushort* table [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = celeg_bf16_to_float(
        table[static_cast<size_t>(token) * width + index]);
}

kernel void celeg_matvec(device const float* weights [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) sum += weights[base + col] * input[col];
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_f16(device const half* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += static_cast<float>(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_bf16(device const ushort* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += celeg_bf16_to_float(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

float celeg_q4_0_value(device const uchar* block, uint column) {
    const uchar packed = block[2 + (column & 15)];
    const uint value = (column & 16) == 0 ? packed & 0x0f : packed >> 4;
    return static_cast<float>(value) - 8.0f;
}

kernel void celeg_matvec_q4_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 18;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * celeg_q4_0_value(block, column) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
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

kernel void celeg_matvec_q5k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 256) * 176;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = column & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q5k_scale_min(block + 4, sub, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

kernel void celeg_matvec_q8_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 34;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * static_cast<float>(static_cast<char>(block[2 + (column & 31)])) *
            input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

float celeg_half_to_float(ushort bits) {
    return static_cast<float>(as_type<half>(bits));
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
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint column = lane; column < cols; column += 32) {
        const uint block_index = column / 256;
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = column & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q4k_scale_min(block + 4, sub, scale, minimum);
        const float value = d * static_cast<float>(scale) *
                            static_cast<float>(celeg_q4k_value(block, within)) -
                            dmin * static_cast<float>(minimum);
        sum += value * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
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
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    for (uint column = lane; column < cols; column += 32) {
        const uint block_index = column / 256;
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 210;
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        const uint within = column & 255;
        const float scale = d * static_cast<float>(
            static_cast<char>(block[192 + within / 16]));
        sum += scale * static_cast<float>(celeg_q6k_value(block, within) - 32) * input[column];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
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

kernel void celeg_embedding_q4_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 18 +
        static_cast<size_t>(index / 32) * 18;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * celeg_q4_0_value(block, index);
}

kernel void celeg_embedding_q5k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 256) * 176 +
        static_cast<size_t>(block_index) * 176;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q5k_scale_min(block + 4, column >> 5, scale, minimum);
    output[index] = d * static_cast<float>(scale) * celeg_q5k_value(block, column) -
        dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q8_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 34 +
        static_cast<size_t>(index / 32) * 34;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * static_cast<float>(static_cast<char>(block[2 + (index & 31)]));
}

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
                           constant uint& page_tokens [[buffer(6)]],
                           uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const size_t page = position / page_tokens;
    const size_t slot = position % page_tokens;
    const size_t offset = (page * page_tokens + slot) * width + index;
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
