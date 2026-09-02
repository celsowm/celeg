// Benchmark-only tensor matmul geometry: keep production's 64 output rows and
// K64 staging, but double the token tile from 128 to 256 so pp512 reuses every
// staged quantized weight tile twice as often. This file is concatenated after
// src/backend/metal/kernels/tensor.metal so it deliberately reuses the
// production decoders (CelegTensorQ4K/CelegTensorQ6K) without duplicating
// quantization semantics.

constant int kCelegTile64x256Rows = 64;
constant int kCelegTile64x256Tokens = 256;
constant int kCelegTile64x256K = 64;
constant int kCelegTile64x256Threads = 128;

template <typename Decoder>
void celeg_matmul_tensor_quantized_64x256(
        device const uchar* weights, device float* input, device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride, uint row_bytes,
        Decoder decoder, threadgroup half* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegTile64x256Rows;
    const int token_offset = static_cast<int>(grid.y) * kCelegTile64x256Tokens;
    const int row_extent = min(kCelegTile64x256Rows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegTile64x256Tokens,
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
        weights_tile, dextents<int32_t, 2>(kCelegTile64x256K, kCelegTile64x256Rows),
        array<int32_t, 2>({1, kCelegTile64x256K}));

    matmul2d<
        matmul2d_descriptor(kCelegTile64x256Tokens, kCelegTile64x256Rows,
                            dynamic_extent, false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegTile64x256K / kCelegBlockValues;
    const int tile_row = static_cast<int>(thread_index) / blocks_per_row;
    const int tile_block = static_cast<int>(thread_index) % blocks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination =
        weights_tile + tile_row * kCelegTile64x256K +
        tile_block * kCelegBlockValues;

    for (int offset = 0; offset < static_cast<int>(cols);
         offset += kCelegTile64x256K) {
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
        const int extent = min(kCelegTile64x256K, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegTile64x256K}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_TENSOR_64X256_QUANTIZED_MATMUL(NAME, DECODER) \
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
    celeg_matmul_tensor_quantized_64x256( \
        weights, input, output, rows, cols, output_rows, output_stride, row_bytes, \
        DECODER{}, weights_tile, thread_index, grid); \
}

CELEG_TENSOR_64X256_QUANTIZED_MATMUL(celeg_matmul_tensor_q4k_64x256, CelegTensorQ4K)
CELEG_TENSOR_64X256_QUANTIZED_MATMUL(celeg_matmul_tensor_q6k_64x256, CelegTensorQ6K)

#undef CELEG_TENSOR_64X256_QUANTIZED_MATMUL
