// Benchmark-only Q4_K TensorOps path combining two strict optimizations:
// 1) stage two production K64 slices before synchronizing, and
// 2) use static M/N/K tensor extents for the aligned full-tile hot path.
// The two matmul2d runs remain K64 and execute in production order, so the
// numerical reduction partition is unchanged. Valid only for dimensions aligned
// to 64 output rows x 128 tokens x K128.

constant int kCelegStaticStageRows = 64;
constant int kCelegStaticStageTokens = 128;
constant int kCelegStaticStageK = 128;
constant int kCelegStaticStageComputeK = 64;
constant int kCelegStaticStageThreads = 128;
constant int kCelegStaticStageBlock = 32;

kernel void celeg_matmul_tensor_q4k_static_stage128(
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
    if ((rows % kCelegStaticStageTokens) != 0u ||
        (cols % kCelegStaticStageK) != 0u ||
        (output_rows % kCelegStaticStageRows) != 0u) {
        return;
    }

    const int row_offset = static_cast<int>(grid.x) * kCelegStaticStageRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegStaticStageTokens;

    auto input_tensor = tensor(
        input, dextents<int, 2>(static_cast<int>(cols), static_cast<int>(rows)),
        array<int, 2>({1, static_cast<int>(cols)}));
    auto output_tensor = tensor(
        output, dextents<int, 2>(static_cast<int>(output_rows), static_cast<int>(rows)),
        array<int, 2>({1, static_cast<int>(output_stride)}));
    auto output_tile = output_tensor.slice<kCelegStaticStageRows,
                                           kCelegStaticStageTokens>(
        row_offset, token_offset);

    auto first_weights = tensor(
        weights_tile,
        extents<int, kCelegStaticStageComputeK, kCelegStaticStageRows>(),
        array<int, 2>({1, kCelegStaticStageK}));
    auto second_weights = tensor(
        weights_tile + kCelegStaticStageComputeK,
        extents<int, kCelegStaticStageComputeK, kCelegStaticStageRows>(),
        array<int, 2>({1, kCelegStaticStageK}));

    constexpr auto descriptor = matmul2d_descriptor(
        kCelegStaticStageTokens, kCelegStaticStageRows,
        kCelegStaticStageComputeK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    auto initial_input =
        input_tensor.slice<kCelegStaticStageComputeK, kCelegStaticStageTokens>(
            0, token_offset);
    auto result = operation.get_destination_cooperative_tensor<
        decltype(initial_input), decltype(first_weights), float>();
    #pragma unroll
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegStaticStageK / kCelegStaticStageBlock;
    constexpr int work_items = kCelegStaticStageRows * blocks_per_row;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegStaticStageK) {
        for (int work = static_cast<int>(thread_index);
             work < work_items; work += kCelegStaticStageThreads) {
            const int tile_row = work / blocks_per_row;
            const int tile_block = work % blocks_per_row;
            const int source_row = row_offset + tile_row;
            const int source_col = offset + tile_block * kCelegStaticStageBlock;
            threadgroup half* destination = weights_tile +
                tile_row * kCelegStaticStageK + tile_block * kCelegStaticStageBlock;
            CelegTensorQ4K{}.store(
                destination,
                weights + static_cast<size_t>(source_row) * row_bytes,
                static_cast<uint>(source_col));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto first_input =
            input_tensor.slice<kCelegStaticStageComputeK, kCelegStaticStageTokens>(
                offset, token_offset);
        operation.run(first_input, first_weights, result);

        auto second_input =
            input_tensor.slice<kCelegStaticStageComputeK, kCelegStaticStageTokens>(
                offset + kCelegStaticStageComputeK, token_offset);
        operation.run(second_input, second_weights, result);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    result.store(output_tile);
}
