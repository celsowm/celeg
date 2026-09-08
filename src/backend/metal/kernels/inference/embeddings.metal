/**
 * @brief embeddings — all celeg_embedding_* kernels.
 */
kernel void celeg_embedding_f16(device const half* table [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = static_cast<float>(
        table[static_cast<size_t>(token) * width + index]);
}



kernel void celeg_embedding_bf16(device const ushort* table [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = celeg_bf16_to_float(
        table[static_cast<size_t>(token) * width + index]);
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
