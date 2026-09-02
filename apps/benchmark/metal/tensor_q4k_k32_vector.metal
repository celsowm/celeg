// Benchmark-only Q4_K TensorOps kernel: 64x128xK32, 128 threads, with each
// thread dequantizing one contiguous 16-value Q4_K chunk through vector loads
// and vector arithmetic before a vector store to the 4 KiB threadgroup tile.
// The scalar formula is deliberately the same as Celeg production:
//   half((d * scale) * q - dmin * minimum)
// so the A/B gate can require bit-identical output.

constant int kCelegK32VectorTileRows = 64;
constant int kCelegK32VectorTileTokens = 128;
constant int kCelegK32VectorTileK = 32;
constant int kCelegK32VectorThreads = 128;
constant int kCelegK32VectorChunk = 16;

inline void celeg_q4k_store16_vector(threadgroup half* destination,
                                     device const uchar* row_data,
                                     uint column) {
    const device uchar* block = row_data + static_cast<size_t>(column / 256u) * 144u;
    const uint within = column & 255u;
    const uint sub = within >> 5;
    const uint index_base = within & 31u;
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
    const device uchar* qs = block + 16 + (sub >> 1) * 32 + index_base;
    const bool high = (sub & 1u) != 0u;

    #pragma unroll full
    for (uint group = 0; group < 4u; ++group) {
        const uchar4 packed = *reinterpret_cast<device const uchar4*>(qs + group * 4u);
        const uint4 raw = uint4(packed);
        const uint4 quantized = high ? (raw >> 4u) : (raw & uint4(0x0fu));
        const float4 decoded = factor * float4(quantized) - float4(bias);
        *reinterpret_cast<threadgroup half4*>(destination + group * 4u) = half4(decoded);
    }
}

kernel void celeg_matmul_tensor_q4k_k32_vector(
        device const uchar* weights [[buffer(0)]],
        device float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        constant uint& row_bytes [[buffer(7)]],
        threadgroup half* weights_tile [[threadgroup(0)]],
        uint thread_index [[thread_index_in_threadgroup]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    const int row_offset = static_cast<int>(grid.x) * kCelegK32VectorTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegK32VectorTileTokens;
    const int row_extent = min(kCelegK32VectorTileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegK32VectorTileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegK32VectorTileK, kCelegK32VectorTileRows),
        array<int32_t, 2>({1, kCelegK32VectorTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegK32VectorTileTokens, kCelegK32VectorTileRows,
                            dynamic_extent, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto first_input = tensor(
        input + static_cast<size_t>(token_offset) * cols,
        dextents<int32_t, 2>(min(kCelegK32VectorTileK, static_cast<int>(cols)),
                             token_extent),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto result = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights_type), float>();
    #pragma unroll full
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int chunks_per_row = kCelegK32VectorTileK / kCelegK32VectorChunk;
    const int tile_row = static_cast<int>(thread_index) / chunks_per_row;
    const int tile_chunk = static_cast<int>(thread_index) % chunks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination = weights_tile +
        tile_row * kCelegK32VectorTileK + tile_chunk * kCelegK32VectorChunk;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegK32VectorTileK) {
        const int source_column = offset + tile_chunk * kCelegK32VectorChunk;
        if (source_row < static_cast<int>(output_rows) &&
            source_column < static_cast<int>(cols)) {
            celeg_q4k_store16_vector(
                destination,
                weights + static_cast<size_t>(source_row) * row_bytes,
                static_cast<uint>(source_column));
        } else {
            #pragma unroll full
            for (int index = 0; index < kCelegK32VectorChunk; ++index) {
                destination[index] = static_cast<half>(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegK32VectorTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegK32VectorTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}
