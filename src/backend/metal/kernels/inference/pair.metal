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
