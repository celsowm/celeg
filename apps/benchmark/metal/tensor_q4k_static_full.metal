// Benchmark-only full-tile Q4_K TensorOps kernel using static tensor extents.
// Geometry, K64 accumulation order, Q4_K decoder, precision and threadgroup
// memory are identical to production. This candidate is valid only when M, N
// and K are aligned to 64x128x64; the LFM2.5-350M pp128/256/512 hot shapes are.

constant int kCelegStaticRows = 64;
constant int kCelegStaticTokens = 128;
constant int kCelegStaticK = 64;
constant int kCelegStaticThreads = 128;
constant int kCelegStaticBlock = 32;

kernel void celeg_matmul_tensor_q4k_static_full(
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
    if ((rows % kCelegStaticTokens) != 0u ||
        (cols % kCelegStaticK) != 0u ||
        (output_rows % kCelegStaticRows) != 0u) {
        return;
    }

    const int row_offset = static_cast<int>(grid.x) * kCelegStaticRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegStaticTokens;

    auto input_tensor = tensor(
        input, dextents<int, 2>(static_cast<int>(cols), static_cast<int>(rows)),
        array<int, 2>({1, static_cast<int>(cols)}));
    auto output_tensor = tensor(
        output, dextents<int, 2>(static_cast<int>(output_rows), static_cast<int>(rows)),
        array<int, 2>({1, static_cast<int>(output_stride)}));
    auto output_tile = output_tensor.slice<kCelegStaticRows, kCelegStaticTokens>(
        row_offset, token_offset);
    auto weights_tensor = tensor(
        weights_tile, extents<int, kCelegStaticK, kCelegStaticRows>(),
        array<int, 2>({1, kCelegStaticK}));

    constexpr auto descriptor = matmul2d_descriptor(
        kCelegStaticTokens, kCelegStaticRows, kCelegStaticK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    auto initial_input = input_tensor.slice<kCelegStaticK, kCelegStaticTokens>(
        0, token_offset);
    auto result = operation.get_destination_cooperative_tensor<
        decltype(initial_input), decltype(weights_tensor), float>();
    #pragma unroll
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegStaticK / kCelegStaticBlock;
    const int tile_row = static_cast<int>(thread_index) / blocks_per_row;
    const int tile_block = static_cast<int>(thread_index) % blocks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination = weights_tile +
        tile_row * kCelegStaticK + tile_block * kCelegStaticBlock;

    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegStaticK) {
        const int source_column = offset + tile_block * kCelegStaticBlock;
        CelegTensorQ4K{}.store(
            destination,
            weights + static_cast<size_t>(source_row) * row_bytes,
            static_cast<uint>(source_column));

        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto input_slice = input_tensor.slice<kCelegStaticK, kCelegStaticTokens>(
            offset, token_offset);
        operation.run(input_slice, weights_tensor, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    result.store(output_tile);
}
