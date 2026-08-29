#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <typename T>
kernel void celeg_matmul_tensor(
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
    constexpr int tile_rows = 64;
    constexpr int tile_tokens = 128;
    constexpr int tile_k = 32;

    const int row_offset = static_cast<int>(grid.x) * tile_rows;
    const int token_offset = static_cast<int>(grid.y) * tile_tokens;
    const int row_extent = min(tile_rows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(tile_tokens, static_cast<int>(rows) - token_offset);
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
        weights_tile, dextents<int32_t, 2>(tile_k, tile_rows),
        array<int32_t, 2>({1, tile_k}));

    matmul2d<
        matmul2d_descriptor(tile_tokens, tile_rows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }
    for (int offset = 0; offset < static_cast<int>(cols); offset += tile_k) {
        for (int index = static_cast<int>(thread_index); index < tile_rows * tile_k;
             index += 128) {
            const int row = index / tile_k;
            const int k = index % tile_k;
            const int source_row = row_offset + row;
            const int source_col = offset + k;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_col < static_cast<int>(cols)
                ? weights[static_cast<size_t>(source_row) * cols + source_col]
                : static_cast<T>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(tile_k, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, tile_k}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

template [[host_name("celeg_matmul_tensor_f16")]]
kernel void celeg_matmul_tensor<half>(
        device const half*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup half*, uint, uint2);

template [[host_name("celeg_matmul_tensor_bf16")]]
kernel void celeg_matmul_tensor<bfloat>(
        device const bfloat*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup bfloat*, uint, uint2);
