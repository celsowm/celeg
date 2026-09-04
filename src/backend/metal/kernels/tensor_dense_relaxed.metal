/**
 * @brief Dense TensorOps fast path with specialized prompt tiles.
 *
 * The regular Fast kernel uses the llama-style K32 staging geometry validated
 * on the real LFM2.5 dense shapes. Each thread moves one contiguous 16-value
 * weight chunk as two raw uint4 vectors, avoiding scalar staging while
 * preserving the exact F16/BF16 bit pattern. The small-prompt N32 kernels keep
 * the previous K64 scalar staging until they are benchmarked independently.
 */

constant int kCelegDenseFastTileK = 32;
constant int kCelegDenseFastChunk = 16;
constant int kCelegDenseFastChunksPerRow =
    kCelegDenseFastTileK / kCelegDenseFastChunk;

template <typename T, int TileTokens>
void celeg_matmul_tensor_dense_relaxed_scalar_impl(
        device const T* weights,
        device float* input,
        device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride,
        threadgroup T* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegTileRows;
    const int token_offset = static_cast<int>(grid.y) * TileTokens;
    const int row_extent = min(kCelegTileRows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(TileTokens, static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegTileK, kCelegTileRows),
        array<int32_t, 2>({1, kCelegTileK}));

    matmul2d<
        matmul2d_descriptor(TileTokens, kCelegTileRows, dynamic_extent,
                            false, true, true,
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

    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegTileK) {
        for (int index = static_cast<int>(thread_index);
             index < kCelegTileRows * kCelegTileK; index += kCelegTileThreads) {
            const int row = index / kCelegTileK;
            const int k = index % kCelegTileK;
            const int source_row = row_offset + row;
            const int source_col = offset + k;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_col < static_cast<int>(cols)
                ? weights[static_cast<size_t>(source_row) * cols + source_col]
                : static_cast<T>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

template <typename T>
void celeg_matmul_tensor_dense_relaxed_stage16_impl(
        device const T* weights,
        device float* input,
        device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride,
        threadgroup T* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegTileTokens;
    const int row_extent = min(kCelegTileRows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegTileTokens, static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegDenseFastTileK, kCelegTileRows),
        array<int32_t, 2>({1, kCelegDenseFastTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegTileTokens, kCelegTileRows, dynamic_extent,
                            false, true, true,
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

    constexpr int work_items = kCelegTileRows * kCelegDenseFastChunksPerRow;
    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegDenseFastTileK) {
        for (int work = static_cast<int>(thread_index);
             work < work_items;
             work += kCelegTileThreads) {
            const int row = work / kCelegDenseFastChunksPerRow;
            const int chunk = work % kCelegDenseFastChunksPerRow;
            const int source_row = row_offset + row;
            const int source_col = offset + chunk * kCelegDenseFastChunk;
            const int destination =
                row * kCelegDenseFastTileK + chunk * kCelegDenseFastChunk;

            if (source_row < static_cast<int>(output_rows) &&
                source_col + kCelegDenseFastChunk <= static_cast<int>(cols)) {
                device const uint4* source_bits =
                    reinterpret_cast<device const uint4*>(weights +
                        static_cast<size_t>(source_row) * cols + source_col);
                threadgroup uint4* destination_bits =
                    reinterpret_cast<threadgroup uint4*>(weights_tile + destination);
                destination_bits[0] = source_bits[0];
                destination_bits[1] = source_bits[1];
            } else {
                #pragma unroll
                for (int index = 0; index < kCelegDenseFastChunk; ++index) {
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
        const int extent = min(kCelegDenseFastTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegDenseFastTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_RELAXED_DENSE_STAGE16_MATMUL(NAME, TYPE) \
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
    celeg_matmul_tensor_dense_relaxed_stage16_impl<TYPE>( \
        weights, input, output, rows, cols, output_rows, output_stride, \
        weights_tile, thread_index, grid); \
}

#define CELEG_RELAXED_DENSE_N32_MATMUL(NAME, TYPE) \
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
    celeg_matmul_tensor_dense_relaxed_scalar_impl<TYPE, 32>( \
        weights, input, output, rows, cols, output_rows, output_stride, \
        weights_tile, thread_index, grid); \
}

CELEG_RELAXED_DENSE_STAGE16_MATMUL(celeg_matmul_tensor_f16_fast, half)
CELEG_RELAXED_DENSE_STAGE16_MATMUL(celeg_matmul_tensor_bf16_fast, bfloat)
CELEG_RELAXED_DENSE_N32_MATMUL(celeg_matmul_tensor_f16_fast_n32, half)
CELEG_RELAXED_DENSE_N32_MATMUL(celeg_matmul_tensor_bf16_fast_n32, bfloat)

#undef CELEG_RELAXED_DENSE_STAGE16_MATMUL
#undef CELEG_RELAXED_DENSE_N32_MATMUL
