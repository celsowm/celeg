// Benchmark-only one-time Q4_K -> F16 materialization for prefill. The
// dequantization arithmetic intentionally matches CelegTensorQ4K::store so the
// dense F16 TensorOps path can be checked bit-for-bit against the native Q4_K
// path. The predecode dispatch is excluded from steady-state matmul timing.

inline void celeg_q4k_predecode32(device half* destination,
                                  device const uchar* row_data,
                                  uint column) {
    const device uchar* block = row_data + static_cast<size_t>(column / 256u) * 144u;
    const uint within = column & 255u;
    const uint sub = within >> 5;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_tensor_scale_min(block + 4, sub, scale, minimum);
    const ushort d_bits = static_cast<ushort>(
        static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
    const float d = static_cast<float>(as_type<half>(d_bits));
    const ushort dmin_bits = static_cast<ushort>(
        static_cast<uint>(block[2]) | (static_cast<uint>(block[3]) << 8));
    const float dmin = static_cast<float>(as_type<half>(dmin_bits));
    const float factor = d * static_cast<float>(scale);
    const float bias = dmin * static_cast<float>(minimum);
    const device uchar* qs = block + 16 + (sub >> 1) * 32;
    const bool high = (sub & 1u) != 0u;
    #pragma unroll
    for (uint index = 0; index < 32u; ++index) {
        const uint packed = qs[index];
        const uint value = high ? (packed >> 4) : (packed & 0x0fu);
        destination[index] =
            static_cast<half>(factor * static_cast<float>(value) - bias);
    }
}

kernel void celeg_predecode_q4k_f16(
        device const uchar* weights [[buffer(0)]],
        device half* dense_weights [[buffer(1)]],
        constant uint& output_rows [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        constant uint& row_bytes [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
    const uint blocks32_per_row = cols / 32u;
    const uint total = output_rows * blocks32_per_row;
    if (index >= total) return;
    const uint row = index / blocks32_per_row;
    const uint block32 = index - row * blocks32_per_row;
    const uint column = block32 * 32u;
    celeg_q4k_predecode32(
        dense_weights + static_cast<size_t>(row) * cols + column,
        weights + static_cast<size_t>(row) * row_bytes,
        column);
}
