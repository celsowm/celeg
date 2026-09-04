// Benchmark-only dense TensorOps candidate matching llama.cpp's dense weight staging.
// Geometry: 64 output rows x 128 prompt tokens x K32, relaxed precision.
// Each of the 128 threads loads one contiguous 16-element weight chunk into
// threadgroup memory instead of staging scalar elements independently.

constant int kCelegDenseStage16TileRows = 64;
constant int kCelegDenseStage16TileTokens = 128;
constant int kCelegDenseStage16TileK = 32;
constant int kCelegDenseStage16TileThreads = 128;
constant int kCelegDenseStage16Chunk = 16;
constant int kCelegDenseStage16ChunksPerRow =
    kCelegDenseStage16TileK / kCelegDenseStage16Chunk;

template <typename T, typename Packed>
void celeg_matmul_tensor_dense_stage16_impl(
        device const T* weights,
        device float* input,
        device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride,
        threadgroup T* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegDenseStage16TileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegDenseStage16TileTokens;
    const int row_extent = min(kCelegDenseStage16TileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegDenseStage16TileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
        weights_tile,
        dextents<int32_t, 2>(kCelegDenseStage16TileK,
                             kCelegDenseStage16TileRows),
        array<int32_t, 2>({1, kCelegDenseStage16TileK}));

    matmul2d<
        matmul2d_descriptor(kCelegDenseStage16TileTokens,
                            kCelegDenseStage16TileRows,
                            dynamic_extent, false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto input_shape = tensor(
        input, dextents<int32_t, 2>(static_cast<int32_t>(cols),
                                    static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto result = operation.template get_destination_cooperative_tensor<
        decltype(input_shape), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int work_items =
        kCelegDenseStage16TileRows * kCelegDenseStage16ChunksPerRow;
    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegDenseStage16TileK) {
        for (int work = static_cast<int>(thread_index);
             work < work_items;
             work += kCelegDenseStage16TileThreads) {
            const int row = work / kCelegDenseStage16ChunksPerRow;
            const int chunk = work % kCelegDenseStage16ChunksPerRow;
            const int source_row = row_offset + row;
            const int source_col = offset + chunk * kCelegDenseStage16Chunk;
            const int destination =
                row * kCelegDenseStage16TileK +
                chunk * kCelegDenseStage16Chunk;

            if (source_row < static_cast<int>(output_rows) &&
                source_col + kCelegDenseStage16Chunk <= static_cast<int>(cols)) {
                device const Packed* packed_source =
                    (device const Packed*)(weights +
                        static_cast<size_t>(source_row) * cols + source_col);
                const Packed packed = *packed_source;
                #pragma unroll
                for (int index = 0; index < kCelegDenseStage16Chunk; ++index) {
                    weights_tile[destination + index] = packed[index / 4][index % 4];
                }
            } else {
                #pragma unroll
                for (int index = 0; index < kCelegDenseStage16Chunk; ++index) {
                    const int k = source_col + index;
                    weights_tile[destination + index] =
                        source_row < static_cast<int>(output_rows) &&
                        k < static_cast<int>(cols)
                        ? weights[static_cast<size_t>(source_row) * cols + k]
                        : static_cast<T>(0);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegDenseStage16TileK,
                               static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegDenseStage16TileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_DENSE_STAGE16_MATMUL(NAME, TYPE, PACKED) \
kernel void NAME( \
        device const TYPE* weights [[buffer(0)]], \
        device float* input [[buffer(1)]], \
        device float* output [[buffer(2)]], \
        constant uint& rows [[buffer(3)]], \
        constant uint& cols [[buffer(4)]], \
        constant uint& output_rows [[buffer(5)]], \
        constant uint& output_stride [[buffer(6)]], \
        threadgroup TYPE* weights_tile [[threadgroup(0)]], \
        uint thread_index [[thread_index_in_threadgroup]], \
        uint2 grid [[threadgroup_position_in_grid]]) { \
    celeg_matmul_tensor_dense_stage16_impl<TYPE, PACKED>( \
        weights, input, output, rows, cols, output_rows, output_stride, \
        weights_tile, thread_index, grid); \
}

CELEG_DENSE_STAGE16_MATMUL(celeg_matmul_tensor_f16_fast_stage16, half, half4x4)
CELEG_DENSE_STAGE16_MATMUL(celeg_matmul_tensor_bf16_fast_stage16, bfloat, bfloat4x4)

#undef CELEG_DENSE_STAGE16_MATMUL
