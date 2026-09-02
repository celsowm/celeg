#pragma once

namespace celeg::metal_benchmark_detail {

inline constexpr const char* kLlamaTileTensorShader = R"CELEG_METAL(

constant int kCelegLlamaTileRows = 128;
constant int kCelegLlamaTileTokens = 256;
constant int kCelegLlamaTileK = 64;
constant int kCelegLlamaTileThreads = 128;

template <typename T>
kernel void celeg_matmul_tensor_llama_tile(
        device const T* weights [[buffer(0)]],
        device float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        threadgroup T* weights_tile [[threadgroup(0)]],
        uint thread_index [[thread_index_in_threadgroup]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    const int row_offset = static_cast<int>(grid.x) * kCelegLlamaTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegLlamaTileTokens;
    const int row_extent = min(kCelegLlamaTileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegLlamaTileTokens,
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
    auto input_tile = input_tensor.slice(0, token_offset);
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(kCelegLlamaTileK, kCelegLlamaTileRows),
        array<int32_t, 2>({1, kCelegLlamaTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegLlamaTileTokens, kCelegLlamaTileRows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegLlamaTileK) {
        for (int index = static_cast<int>(thread_index);
             index < kCelegLlamaTileRows * kCelegLlamaTileK;
             index += kCelegLlamaTileThreads) {
            const int row = index / kCelegLlamaTileK;
            const int k = index % kCelegLlamaTileK;
            const int source_row = row_offset + row;
            const int source_col = offset + k;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_col < static_cast<int>(cols)
                ? weights[static_cast<size_t>(source_row) * cols + source_col]
                : static_cast<T>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegLlamaTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegLlamaTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

template [[host_name("celeg_matmul_tensor_llama_f16")]]
kernel void celeg_matmul_tensor_llama_tile<half>(
        device const half*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup half*, uint, uint2);

template [[host_name("celeg_matmul_tensor_llama_bf16")]]
kernel void celeg_matmul_tensor_llama_tile<bfloat>(
        device const bfloat*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup bfloat*, uint, uint2);

template <typename Decoder>
void celeg_matmul_tensor_quantized_llama_tile(
        device const uchar* weights, device float* input, device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride, uint row_bytes,
        Decoder decoder, threadgroup half* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegLlamaTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegLlamaTileTokens;
    const int row_extent = min(kCelegLlamaTileRows,
                               static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegLlamaTileTokens,
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
        weights_tile, dextents<int32_t, 2>(kCelegLlamaTileK, kCelegLlamaTileRows),
        array<int32_t, 2>({1, kCelegLlamaTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegLlamaTileTokens, kCelegLlamaTileRows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegLlamaTileK / kCelegBlockValues;
    constexpr int work_items = kCelegLlamaTileRows * blocks_per_row;
    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegLlamaTileK) {
        for (int work = static_cast<int>(thread_index); work < work_items;
             work += kCelegLlamaTileThreads) {
            const int tile_row = work / blocks_per_row;
            const int tile_block = work % blocks_per_row;
            const int source_row = row_offset + tile_row;
            const int source_column = offset + tile_block * kCelegBlockValues;
            threadgroup half* destination = weights_tile +
                tile_row * kCelegLlamaTileK + tile_block * kCelegBlockValues;
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
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(kCelegLlamaTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegLlamaTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(NAME, DECODER) \
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
    celeg_matmul_tensor_quantized_llama_tile( \
        weights, input, output, rows, cols, output_rows, output_stride, row_bytes, \
        DECODER{}, weights_tile, thread_index, grid); \
}

CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(celeg_matmul_tensor_llama_q4_0, CelegTensorQ4_0)
CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(celeg_matmul_tensor_llama_q8_0, CelegTensorQ8_0)
CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(celeg_matmul_tensor_llama_q4k, CelegTensorQ4K)
CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(celeg_matmul_tensor_llama_q5k, CelegTensorQ5K)
CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL(celeg_matmul_tensor_llama_q6k, CelegTensorQ6K)

#undef CELEG_TENSOR_LLAMA_QUANTIZED_MATMUL

)CELEG_METAL";

}  // namespace celeg::metal_benchmark_detail
