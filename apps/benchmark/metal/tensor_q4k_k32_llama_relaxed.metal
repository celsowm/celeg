// Benchmark-only reduced-precision Q4_K TensorOps kernel matching the current
// llama.cpp mul_mm decomposition: 64 output rows x 128 tokens x K32, 128
// threads, one 16-value dequantization chunk per thread. This deliberately
// enables relaxed_precision=true so it can be compared with Celeg's opt-in
// reduced-precision K64 production path.

constant int kCelegLlamaRelaxedK32TileRows = 64;
constant int kCelegLlamaRelaxedK32TileTokens = 128;
constant int kCelegLlamaRelaxedK32TileK = 32;
constant int kCelegLlamaRelaxedK32Threads = 128;
constant int kCelegLlamaRelaxedK32Chunk = 16;

inline uchar2 celeg_llama_relaxed_scale_min_k4(int j, int k, device const uchar* q) {
    return j < 4
        ? uchar2{uchar(q[j + k] & 63u), uchar(q[j + 4 + k] & 63u)}
        : uchar2{
              uchar((q[j + 4 + k] & 0x0fu) | ((q[j - 4 + k] & 0xc0u) >> 2)),
              uchar((q[j + 4 + k] >> 4) | ((q[j + k] & 0xc0u) >> 2))};
}

inline void celeg_q4k_store16_llama_relaxed(threadgroup half* destination,
                                             device const uchar* row_data,
                                             uint column) {
    const device uchar* block = row_data + static_cast<size_t>(column / 256u) * 144u;
    short il = static_cast<short>((column & 255u) / 16u);
    short scale_index = static_cast<short>((il / 4) * 2);
    const device uchar* q = block + 16 + (il / 4) * 32 + 16 * (il & 1);
    il &= 3;
    const uchar2 sc = celeg_llama_relaxed_scale_min_k4(scale_index, il / 2, block + 4);

    const ushort d_bits = static_cast<ushort>(
        static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
    const float d_all = static_cast<float>(as_type<half>(d_bits));
    const ushort dmin_bits = static_cast<ushort>(
        static_cast<uint>(block[2]) | (static_cast<uint>(block[3]) << 8));
    const float minimum_all = static_cast<float>(as_type<half>(dmin_bits));

    const float d = il < 2 ? d_all : d_all / 16.0f;
    const float dl = d * static_cast<float>(sc[0]);
    const float ml = minimum_all * static_cast<float>(sc[1]);
    const uint mask = il < 2 ? 0x0fu : 0xf0u;

    #pragma unroll
    for (uint index = 0; index < kCelegLlamaRelaxedK32Chunk; ++index) {
        destination[index] = static_cast<half>(
            dl * static_cast<float>(static_cast<uint>(q[index]) & mask) - ml);
    }
}

kernel void celeg_matmul_tensor_q4k_k32_llama_relaxed(
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
    const int row_offset = static_cast<int>(grid.x) * kCelegLlamaRelaxedK32TileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegLlamaRelaxedK32TileTokens;
    const int row_extent = min(kCelegLlamaRelaxedK32TileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegLlamaRelaxedK32TileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile,
        dextents<int32_t, 2>(kCelegLlamaRelaxedK32TileK, kCelegLlamaRelaxedK32TileRows),
        array<int32_t, 2>({1, kCelegLlamaRelaxedK32TileK}));

    matmul2d<
        matmul2d_descriptor(kCelegLlamaRelaxedK32TileTokens,
                            kCelegLlamaRelaxedK32TileRows,
                            dynamic_extent, false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto first_input = tensor(
        input + static_cast<size_t>(token_offset) * cols,
        dextents<int32_t, 2>(min(kCelegLlamaRelaxedK32TileK, static_cast<int>(cols)),
                             token_extent),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto result = operation.get_destination_cooperative_tensor<
        decltype(first_input), decltype(weights_type), float>();
    #pragma unroll
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int chunks_per_row =
        kCelegLlamaRelaxedK32TileK / kCelegLlamaRelaxedK32Chunk;
    const int tile_row = static_cast<int>(thread_index) / chunks_per_row;
    const int tile_chunk = static_cast<int>(thread_index) % chunks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination = weights_tile +
        tile_row * kCelegLlamaRelaxedK32TileK + tile_chunk * kCelegLlamaRelaxedK32Chunk;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegLlamaRelaxedK32TileK) {
        const int source_column = offset + tile_chunk * kCelegLlamaRelaxedK32Chunk;
        if (source_row < static_cast<int>(output_rows) &&
            source_column < static_cast<int>(cols)) {
            celeg_q4k_store16_llama_relaxed(
                destination,
                weights + static_cast<size_t>(source_row) * row_bytes,
                static_cast<uint>(source_column));
        } else {
            #pragma unroll
            for (int index = 0; index < kCelegLlamaRelaxedK32Chunk; ++index) {
                destination[index] = static_cast<half>(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegLlamaRelaxedK32TileK,
                               static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegLlamaRelaxedK32TileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}
