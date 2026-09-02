// Benchmark-only Q4_K TensorOps path that stages two production K64 slices at
// once. The matmul operation itself remains strict K64 and is invoked twice in
// the same order as production, preserving the numerical reduction partition.
// This changes only synchronization: one decode128/barrier pair feeds two K64
// runs before the tile is reused.

constant int kCelegStage128Rows = 64;
constant int kCelegStage128Tokens = 128;
constant int kCelegStage128K = 128;
constant int kCelegStage128ComputeK = 64;
constant int kCelegStage128Threads = 128;
constant int kCelegStage128Block = 32;

kernel void celeg_matmul_tensor_q4k_stage128(
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
    const int row_offset = static_cast<int>(grid.x) * kCelegStage128Rows;
    const int token_offset = static_cast<int>(grid.y) * kCelegStage128Tokens;
    const int row_extent = min(kCelegStage128Rows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegStage128Tokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);

    auto initial_input = tensor(
        input + static_cast<size_t>(token_offset) * cols,
        dextents<int32_t, 2>(min(kCelegStage128ComputeK,
                                 static_cast<int>(cols)), token_extent),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto initial_weights = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile,
        dextents<int32_t, 2>(min(kCelegStage128ComputeK,
                                 static_cast<int>(cols)), row_extent),
        array<int32_t, 2>({1, kCelegStage128K}));

    matmul2d<
        matmul2d_descriptor(kCelegStage128Tokens, kCelegStage128Rows,
                            dynamic_extent, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(initial_input), decltype(initial_weights), float>();
    #pragma unroll
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegStage128K / kCelegStage128Block;
    constexpr int work_items = kCelegStage128Rows * blocks_per_row;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegStage128K) {
        for (int work = static_cast<int>(thread_index);
             work < work_items; work += kCelegStage128Threads) {
            const int tile_row = work / blocks_per_row;
            const int tile_block = work % blocks_per_row;
            const int source_row = row_offset + tile_row;
            const int source_col = offset + tile_block * kCelegStage128Block;
            threadgroup half* destination = weights_tile +
                tile_row * kCelegStage128K + tile_block * kCelegStage128Block;
            if (source_row < static_cast<int>(output_rows) &&
                source_col < static_cast<int>(cols)) {
                CelegTensorQ4K{}.store(
                    destination,
                    weights + static_cast<size_t>(source_row) * row_bytes,
                    static_cast<uint>(source_col));
            } else {
                #pragma unroll
                for (int index = 0; index < kCelegStage128Block; ++index) {
                    destination[index] = static_cast<half>(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int first_extent = min(kCelegStage128ComputeK,
                                     static_cast<int>(cols) - offset);
        auto first_input = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(first_extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto first_weights = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(first_extent, row_extent),
            array<int32_t, 2>({1, kCelegStage128K}));
        operation.run(first_input, first_weights, result);

        const int second_offset = offset + kCelegStage128ComputeK;
        if (second_offset < static_cast<int>(cols)) {
            const int second_extent = min(kCelegStage128ComputeK,
                                          static_cast<int>(cols) - second_offset);
            auto second_input = tensor(
                input + static_cast<size_t>(token_offset) * cols + second_offset,
                dextents<int32_t, 2>(second_extent, token_extent),
                array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
            auto second_weights = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
                weights_tile + kCelegStage128ComputeK,
                dextents<int32_t, 2>(second_extent, row_extent),
                array<int32_t, 2>({1, kCelegStage128K}));
            operation.run(second_input, second_weights, result);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}
