// Optional quantized TensorOps fast path using Metal MPP reduced precision.
//
// These kernels intentionally set matmul2d_descriptor::relaxed_precision to
// true. They are therefore not bit-exact with the default Celeg path and must
// only be selected by the explicit CELEG_METAL_TENSOR_RELAXED_PRECISION=1
// runtime opt-in. The decoder and K/output-row geometry are shared across the
// N128 throughput tile and N32 short-prefill tile.

constant int kCelegRelaxedTileRows = 64;
constant int kCelegRelaxedTileTokens = 128;
constant int kCelegRelaxedSmallTileTokens = 32;
constant int kCelegRelaxedTileK = 64;
constant int kCelegRelaxedTileThreads = 128;

template <typename Decoder, int TileTokens>
void celeg_matmul_tensor_quantized_relaxed(
        device const uchar* weights, device float* input, device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride, uint row_bytes,
        Decoder decoder, threadgroup half* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegRelaxedTileRows;
    const int token_offset = static_cast<int>(grid.y) * TileTokens;
    const int row_extent = min(kCelegRelaxedTileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(TileTokens,
                                 static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto input_tensor = tensor(
        input, dextents<int32_t, 2>(static_cast<int32_t>(cols),
                                    static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegRelaxedTileK, kCelegRelaxedTileRows),
        array<int32_t, 2>({1, kCelegRelaxedTileK}));

    matmul2d<
        matmul2d_descriptor(TileTokens, kCelegRelaxedTileRows,
                            dynamic_extent, false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegRelaxedTileK / kCelegBlockValues;
    const int tile_row = static_cast<int>(thread_index) / blocks_per_row;
    const int tile_block = static_cast<int>(thread_index) % blocks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination =
        weights_tile + tile_row * kCelegRelaxedTileK +
        tile_block * kCelegBlockValues;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegRelaxedTileK) {
        const int source_column = offset + tile_block * kCelegBlockValues;
        if (source_row < static_cast<int>(output_rows) &&
            source_column < static_cast<int>(cols)) {
            decoder.store(destination,
                          weights + static_cast<size_t>(source_row) * row_bytes,
                          static_cast<uint>(source_column));
        } else {
            for (int index = 0; index < kCelegBlockValues; ++index) {
                destination[index] = static_cast<half>(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegRelaxedTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegRelaxedTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_RELAXED_QUANTIZED_MATMUL(NAME, DECODER, TILE_TOKENS) \
kernel void NAME( \
        device const uchar* weights [[buffer(0)]], \
        device float* input [[buffer(1)]], \
        device float* output [[buffer(2)]], \
        constant uint& rows [[buffer(3)]], \
        constant uint& cols [[buffer(4)]], \
        constant uint& output_rows [[buffer(5)]], \
        constant uint& output_stride [[buffer(6)]], \
        constant uint& row_bytes [[buffer(7)]], \
        threadgroup half* weights_tile [[threadgroup(0)]], \
        uint thread_index [[thread_index_in_threadgroup]], \
        uint2 grid [[threadgroup_position_in_grid]]) { \
    celeg_matmul_tensor_quantized_relaxed<DECODER, TILE_TOKENS>( \
        weights, input, output, rows, cols, output_rows, output_stride, row_bytes, \
        DECODER{}, weights_tile, thread_index, grid); \
}

CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q4_0_relaxed, CelegTensorQ4_0, 128)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q4k_relaxed, CelegTensorQ4K, 128)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q5k_relaxed, CelegTensorQ5K, 128)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q6k_relaxed, CelegTensorQ6K, 128)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q8_0_relaxed, CelegTensorQ8_0, 128)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q4_0_relaxed_n32, CelegTensorQ4_0, 32)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q4k_relaxed_n32, CelegTensorQ4K, 32)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q5k_relaxed_n32, CelegTensorQ5K, 32)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q6k_relaxed_n32, CelegTensorQ6K, 32)
CELEG_RELAXED_QUANTIZED_MATMUL(celeg_matmul_tensor_q8_0_relaxed_n32, CelegTensorQ8_0, 32)

#undef CELEG_RELAXED_QUANTIZED_MATMUL
