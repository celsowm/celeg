kernel void celeg_matmul(device const float* weights [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& output_rows [[buffer(5)]],
                         constant uint& output_stride [[buffer(6)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    float sum = 0.0f;
    const size_t input_base = static_cast<size_t>(token) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += weights[weight_base + col] * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_f16(device const half* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& output_rows [[buffer(5)]],
                             constant uint& output_stride [[buffer(6)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    float sum = 0.0f;
    const size_t input_base = static_cast<size_t>(token) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += static_cast<float>(weights[weight_base + col]) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_bf16(device const ushort* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& output_rows [[buffer(5)]],
                              constant uint& output_stride [[buffer(6)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    float sum = 0.0f;
    const size_t input_base = static_cast<size_t>(token) * cols;
    const size_t weight_base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) {
        sum += celeg_bf16_to_float(weights[weight_base + col]) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_q4_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& output_rows [[buffer(5)]],
                              constant uint& output_stride [[buffer(6)]],
                              constant uint& row_bytes [[buffer(7)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(token) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 18;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * celeg_q4_0_value(block, col) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_q5k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& output_rows [[buffer(5)]],
                             constant uint& output_stride [[buffer(6)]],
                             constant uint& row_bytes [[buffer(7)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(token) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 176;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q5k_scale_min(block + 4, sub, scale, minimum);
        sum += (d * static_cast<float>(scale) * celeg_q5k_value(block, within) -
                dmin * static_cast<float>(minimum)) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_q8_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& output_rows [[buffer(5)]],
                              constant uint& output_stride [[buffer(6)]],
                              constant uint& row_bytes [[buffer(7)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(token) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 32) * 34;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        sum += d * static_cast<float>(static_cast<char>(block[2 + (col & 31)])) *
            input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_q4k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& output_rows [[buffer(5)]],
                             constant uint& output_stride [[buffer(6)]],
                             constant uint& row_bytes [[buffer(7)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(token) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const device uchar* block = row_data + static_cast<size_t>(col / 256) * 144;
        const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                            (static_cast<ushort>(block[1]) << 8));
        const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                               (static_cast<ushort>(block[3]) << 8));
        const uint within = col & 255;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_q4k_scale_min(block + 4, sub, scale, minimum);
        const float value = d * static_cast<float>(scale) *
            static_cast<float>(celeg_q4k_value(block, within)) -
            dmin * static_cast<float>(minimum);
        sum += value * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

kernel void celeg_matmul_q6k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& output_rows [[buffer(5)]],
                             constant uint& output_stride [[buffer(6)]],
                             constant uint& row_bytes [[buffer(7)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint2 grid [[threadgroup_position_in_grid]]) {
    const uint row = grid.x * 8 + simd;
    const uint token = grid.y;
    if (row >= output_rows) return;
    const device uchar* row_data = weights + static_cast<size_t>(row) * row_bytes;
    const size_t input_base = static_cast<size_t>(token) * cols;
    float sum = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        const uint block_index = col / 256;
        const device uchar* block = row_data + static_cast<size_t>(block_index) * 210;
        const uint within = col & 255;
        const uint sub = within >> 6;
        const uint q = static_cast<uint>(celeg_q6k_value(block, within));
        const int scale = static_cast<int>(static_cast<char>(block[192 + within / 16]));
        const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                            (static_cast<ushort>(block[209]) << 8));
        sum += d * static_cast<float>(static_cast<int>(q) - 32) *
            static_cast<float>(scale) * input[input_base + col];
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) output[static_cast<size_t>(token) * output_stride + row] = reduced;
}

