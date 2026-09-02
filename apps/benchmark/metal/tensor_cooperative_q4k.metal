// Benchmark-only Q4_K TensorOps path that dequantizes weights directly into
// the right-input cooperative tensor. Geometry, K partition and precision stay
// at the production values (64 output rows x 128 tokens x K64, strict FP32
// accumulation); only the threadgroup half staging round-trip is removed.

constant int kCelegCoopTileRows = 64;
constant int kCelegCoopTileTokens = 128;
constant int kCelegCoopTileK = 64;

inline half celeg_q4k_value(device const uchar* row_data, uint column) {
    const device uchar* block = row_data + static_cast<size_t>(column / 256u) * 144u;
    const uint within = column & 255u;
    const uint sub = within >> 5;
    const uint index = within & 31u;
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
    const uint packed = qs[index];
    const uint value = (sub & 1u) != 0u ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<half>(factor * static_cast<float>(value) - bias);
}

kernel void celeg_matmul_tensor_q4k_cooperative(
        device const uchar* weights [[buffer(0)]],
        device float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        constant uint& row_bytes [[buffer(7)]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    const int row_offset = static_cast<int>(grid.x) * kCelegCoopTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegCoopTileTokens;
    const int row_extent = min(kCelegCoopTileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegCoopTileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    // Static K64 is intentional: cooperative input layout must be known when
    // the private tensor is created. LFM2.5 linear K dimensions are K64-aligned.
    if ((cols & (kCelegCoopTileK - 1)) != 0u) return;

    matmul2d<
        matmul2d_descriptor(kCelegCoopTileTokens, kCelegCoopTileRows,
                            kCelegCoopTileK, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;

    auto first_input = tensor(
        input + static_cast<size_t>(token_offset) * cols,
        dextents<int32_t, 2>(kCelegCoopTileK, token_extent),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto first_weights =
        operation.get_right_input_cooperative_tensor<float, half, float>();
    auto result = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(first_weights), float>();

    #pragma unroll full
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegCoopTileK) {
        auto weight_fragment =
            operation.get_right_input_cooperative_tensor<float, half, float>();
        #pragma unroll full
        for (uint16_t index = 0; index < weight_fragment.get_capacity(); ++index) {
            if (!weight_fragment.is_valid_element(index)) continue;
            const auto coordinate = weight_fragment.get_multidimensional_index(index);
            const int source_column = offset + coordinate[0];
            const int source_row = row_offset + coordinate[1];
            weight_fragment[index] =
                source_row < static_cast<int>(output_rows)
                    ? celeg_q4k_value(
                          weights + static_cast<size_t>(source_row) * row_bytes,
                          static_cast<uint>(source_column))
                    : static_cast<half>(0);
        }

        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(kCelegCoopTileK, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        operation.run(input_slice, weight_fragment, result);
    }

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    result.store(output_tensor.slice(row_offset, token_offset));
}
