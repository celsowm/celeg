kernel void celeg_matvec_pair(device const float* first [[buffer(0)]],
                              device const float* second [[buffer(1)]],
                              device const float* input [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              constant uint& rows [[buffer(4)]],
                              constant uint& cols [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device float* weights = upper ? second : first;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) sum += weights[base + col] * input[col];
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_f16(device const half* first [[buffer(0)]],
                                  device const half* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device half* weights = upper ? second : first;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += static_cast<float>(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_bf16(device const ushort* first [[buffer(0)]],
                                   device const ushort* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device ushort* weights = upper ? second : first;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += celeg_bf16_to_float(weights[base + col]) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_q4_0(device const uchar* first [[buffer(0)]],
                                   device const uchar* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   constant uint& row_bytes [[buffer(6)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 18;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * celeg_q4_0_value(block, col) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_q5k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& row_bytes [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 176;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q5k_scale_min(block + 4, within >> 5, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_q8_0(device const uchar* first [[buffer(0)]],
                                   device const uchar* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   constant uint& row_bytes [[buffer(6)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 34;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * static_cast<float>(static_cast<char>(block[2 + (col & 31)])) *
            input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matvec_pair_q4k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& row_bytes [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q4k_scale_min(block + 4, within >> 5, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q4k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_residual_matvec_pair_q4k(
    device const uchar* first [[buffer(0)]],
    device const uchar* second [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device const float* norm_weight [[buffer(4)]],
    device float* combined_output [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    constant uint& cols [[buffer(8)]],
    constant uint& row_bytes [[buffer(9)]],
    constant float& multiplier [[buffer(10)]],
    constant float& epsilon [[buffer(11)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    float sum = 0.0f;
    for (uint column = index; column < cols; column += 256u) {
        const float value = input[column] * multiplier + residual[column];
        shared[column] = value;
        combined_output[column] = value;
        sum += value * value;
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) shared[cols + 1u + simd] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index == 0) {
        float total = 0.0f;
        for (uint other = 0; other < 8u; ++other) total += shared[cols + 1u + other];
        shared[cols] = rsqrt(total / static_cast<float>(cols) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = index; column < cols; column += 256u) {
        shared[column] = shared[column] * shared[cols] * norm_weight[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool upper = simd >= 4u;
    const uint row = group * 4u + (simd & 3u);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float dot = 0.0f;
    for (uint block_index = 0; block_index * 256u < cols; ++block_index) {
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 144u;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint base = block_index * 256u;
        for (uint sub = 0; sub < 8u; ++sub) {
            const uint column = base + sub * 32u + lane;
            uchar scale = 0;
            uchar minimum = 0;
            celeg_q4k_scale_min(block + 4, sub, scale, minimum);
            const float weight = d * static_cast<float>(scale) *
                    static_cast<float>(celeg_q4k_value(block, sub * 32u + lane)) -
                dmin * static_cast<float>(minimum);
            dot += weight * shared[column];
        }
    }
    const float result = simd_sum(dot);
    if (lane == 0) output[(upper ? rows : 0) + row] = result;
}

float celeg_residual_q5k_weight(device const uchar* block, uint within) {
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q5k_scale_min(block + 4, within >> 5, scale, minimum);
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    return d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
        dmin * static_cast<float>(minimum);
}

float celeg_residual_q6k_weight(device const uchar* block, uint within) {
    const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                        (static_cast<ushort>(block[209]) << 8));
    return d * static_cast<float>(static_cast<char>(block[192 + within / 16])) *
        static_cast<float>(celeg_q6k_value(block, within) - 32);
}

#define CELEG_RESIDUAL_MATVEC_PAIR_QK(NAME, BLOCK_BYTES, DECODE) \
kernel void celeg_residual_matvec_pair_##NAME( \
    device const uchar* first [[buffer(0)]], device const uchar* second [[buffer(1)]], \
    device const float* input [[buffer(2)]], device const float* residual [[buffer(3)]], \
    device const float* norm_weight [[buffer(4)]], device float* combined_output [[buffer(5)]], \
    device float* output [[buffer(6)]], constant uint& rows [[buffer(7)]], \
    constant uint& cols [[buffer(8)]], constant uint& row_bytes [[buffer(9)]], \
    constant float& multiplier [[buffer(10)]], constant float& epsilon [[buffer(11)]], \
    threadgroup float* shared [[threadgroup(0)]], \
    uint index [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) { \
    float sum = 0.0f; \
    for (uint column = index; column < cols; column += 256u) { \
        const float value = input[column] * multiplier + residual[column]; \
        shared[column] = value; \
        combined_output[column] = value; \
        sum += value * value; \
    } \
    const float reduced = simd_sum(sum); \
    if (lane == 0) shared[cols + 1u + simd] = reduced; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (index == 0) { \
        float total = 0.0f; \
        for (uint other = 0; other < 8u; ++other) total += shared[cols + 1u + other]; \
        shared[cols] = rsqrt(total / static_cast<float>(cols) + epsilon); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint column = index; column < cols; column += 256u) { \
        shared[column] = shared[column] * shared[cols] * norm_weight[column]; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    const bool upper = simd >= 4u; \
    const uint row = group * 4u + (simd & 3u); \
    if (row >= rows) return; \
    const device uchar* matrix = upper ? second : first; \
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes; \
    float dot = 0.0f; \
    for (uint block_index = 0; block_index * 256u < cols; ++block_index) { \
        const device uchar* block = row_data + static_cast<size_t>(block_index) * BLOCK_BYTES; \
        for (uint sub = 0; sub < 8u; ++sub) { \
            const uint column = block_index * 256u + sub * 32u + lane; \
            dot += DECODE(block, sub * 32u + lane) * shared[column]; \
        } \
    } \
    const float result = simd_sum(dot); \
    if (lane == 0) output[(upper ? rows : 0) + row] = result; \
}

CELEG_RESIDUAL_MATVEC_PAIR_QK(q5k, 176u, celeg_residual_q5k_weight)
CELEG_RESIDUAL_MATVEC_PAIR_QK(q6k, 210u, celeg_residual_q6k_weight)

#undef CELEG_RESIDUAL_MATVEC_PAIR_QK

kernel void celeg_matvec_pair_q6k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& row_bytes [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint group [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = group * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const uint within = col & 255;
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 210;
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        const float scale = d * static_cast<float>(static_cast<char>(block[192 + within / 16]));
        sum += scale * static_cast<float>(celeg_q6k_value(block, within) - 32) * input[col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[(upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair(device const float* first [[buffer(0)]],
                              device const float* second [[buffer(1)]],
                              device const float* input [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              constant uint& rows [[buffer(4)]],
                              constant uint& cols [[buffer(5)]],
                              constant uint& output_stride [[buffer(6)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device float* weights = upper ? second : first;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        sum += weights[weight_base + col] * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_f16(device const half* first [[buffer(0)]],
                                  device const half* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& output_stride [[buffer(6)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device half* weights = upper ? second : first;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        sum += static_cast<float>(weights[weight_base + col]) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_bf16(device const ushort* first [[buffer(0)]],
                                   device const ushort* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   constant uint& output_stride [[buffer(6)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device ushort* weights = upper ? second : first;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        sum += celeg_bf16_to_float(weights[weight_base + col]) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_q4_0(device const uchar* first [[buffer(0)]],
                                   device const uchar* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   constant uint& output_stride [[buffer(6)]],
                                   constant uint& row_bytes [[buffer(7)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 18;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * celeg_q4_0_value(block, col) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_q5k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& output_stride [[buffer(6)]],
                                  constant uint& row_bytes [[buffer(7)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 176;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q5k_scale_min(block + 4, within >> 5, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_q8_0(device const uchar* first [[buffer(0)]],
                                   device const uchar* second [[buffer(1)]],
                                   device const float* input [[buffer(2)]],
                                   device float* output [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& cols [[buffer(5)]],
                                   constant uint& output_stride [[buffer(6)]],
                                   constant uint& row_bytes [[buffer(7)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 34;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * static_cast<float>(static_cast<char>(block[2 + (col & 31)])) *
            input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_q4k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& output_stride [[buffer(6)]],
                                  constant uint& row_bytes [[buffer(7)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q4k_scale_min(block + 4, within >> 5, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q4k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}

kernel void celeg_matmul_pair_q6k(device const uchar* first [[buffer(0)]],
                                  device const uchar* second [[buffer(1)]],
                                  device const float* input [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant uint& rows [[buffer(4)]],
                                  constant uint& cols [[buffer(5)]],
                                  constant uint& output_stride [[buffer(6)]],
                                  constant uint& row_bytes [[buffer(7)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint2 grid [[threadgroup_position_in_grid]]) {
    const bool upper = simd >= 4;
    const uint row = grid.x * 4 + (simd & 3);
    if (row >= rows) return;
    const device uchar* matrix = upper ? second : first;
    const device uchar* row_data = matrix + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(grid.y) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const uint within = col & 255;
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 210;
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        const float scale = d * static_cast<float>(static_cast<char>(block[192 + within / 16]));
        sum += scale * static_cast<float>(celeg_q6k_value(block, within) - 32) *
            input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(grid.y) * output_stride +
                          (upper ? rows : 0) + row] = reduced;
}
