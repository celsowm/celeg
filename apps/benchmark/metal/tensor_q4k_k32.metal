// Benchmark-only Q4_K TensorOps path matching the current llama.cpp K-tile
// decomposition: 64 output rows x 128 tokens x K32, 128 threads, one 16-value
// dequantization chunk per thread. Unlike Celeg's historical K32 path, this
// keeps block-wise scale/header decoding instead of per-element LOAD_WEIGHT.

constant int kCelegK32TileRows = 64;
constant int kCelegK32TileTokens = 128;
constant int kCelegK32TileK = 32;
constant int kCelegK32Threads = 128;
constant int kCelegK32Chunk = 16;

inline void celeg_q4k_store16(threadgroup half* destination,
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
    const device uchar* qs = block + 16 + (sub >> 1) * 32;
    const bool high = (sub & 1u) != 0u;
    #pragma unroll
    for (uint index = 0; index < kCelegK32Chunk; ++index) {
        const uint packed = qs[index_base + index];
        const uint value = high ? (packed >> 4) : (packed & 0x0fu);
        destination[index] =
            static_cast<half>(factor * static_cast<float>(value) - bias);
    }
}

kernel void celeg_matmul_tensor_q4k_k32(
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
    const int row_offset = static_cast<int>(grid.x) * kCelegK32TileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegK32TileTokens;
    const int row_extent = min(kCelegK32TileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegK32TileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegK32TileK, kCelegK32TileRows),
        array<int32_t, 2>({1, kCelegK32TileK}));

    matmul2d<
        matmul2d_descriptor(kCelegK32TileTokens, kCelegK32TileRows,
                            dynamic_extent, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto first_input = tensor(
        input + static_cast<size_t>(token_offset) * cols,
        dextents<int32_t, 2>(min(kCelegK32TileK, static_cast<int>(cols)), token_extent),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto result = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights_type), float>();
    #pragma unroll
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int chunks_per_row = kCelegK32TileK / kCelegK32Chunk;
    const int tile_row = static_cast<int>(thread_index) / chunks_per_row;
    const int tile_chunk = static_cast<int>(thread_index) % chunks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination =
        weights_tile + tile_row * kCelegK32TileK + tile_chunk * kCelegK32Chunk;

    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegK32TileK) {
        const int source_column = offset + tile_chunk * kCelegK32Chunk;
        if (source_row < static_cast<int>(output_rows) &&
            source_column < static_cast<int>(cols)) {
            celeg_q4k_store16(destination,
                              weights + static_cast<size_t>(source_row) * row_bytes,
                              static_cast<uint>(source_column));
        } else {
            #pragma unroll
            for (int index = 0; index < kCelegK32Chunk; ++index) {
                destination[index] = static_cast<half>(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegK32TileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegK32TileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}
