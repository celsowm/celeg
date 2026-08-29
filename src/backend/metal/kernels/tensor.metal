#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <typename T>
kernel void celeg_matmul_tensor(
        device T* weights [[buffer(0)]],
        device float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    constexpr int tile_rows = 64;
    constexpr int tile_tokens = 128;
    constexpr int tile_k = 32;

    auto weight_tensor = tensor(
        weights, dextents<int32_t, 2>(static_cast<int32_t>(cols),
                                      static_cast<int32_t>(output_rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto input_tensor = tensor(
        input, dextents<int32_t, 2>(static_cast<int32_t>(cols),
                                    static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
    auto output_tensor = tensor(
        output, dextents<int32_t, 2>(static_cast<int32_t>(output_rows),
                                     static_cast<int32_t>(rows)),
        array<int32_t, 2>({1, static_cast<int32_t>(output_stride)}));

    const int row_offset = static_cast<int>(grid.y) * tile_rows;
    const int token_offset = static_cast<int>(grid.x) * tile_tokens;
    const int row_extent = min(tile_rows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(tile_tokens, static_cast<int>(rows) - token_offset);
    if (row_extent <= 0 || token_extent <= 0) return;

    auto weights_tile = weight_tensor.slice(0, row_offset);
    auto input_tile = input_tensor.slice(0, token_offset);
    auto output_tile = output_tensor.slice(row_offset, token_offset);

    matmul2d<
        matmul2d_descriptor(tile_tokens, tile_rows, dynamic_extent,
                            false, true, true,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;

    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weights_tile), float>();
    for (int offset = 0; offset < static_cast<int>(cols); offset += tile_k) {
        const int extent = min(tile_k, static_cast<int>(cols) - offset);
        auto input_slice = input_tile.slice(offset, 0);
        auto weights_slice = weights_tile.slice(offset, 0);
        operation.run(input_slice, weights_slice, result);
        if (extent < tile_k) break;
    }
    result.store(output_tile);
}

template [[host_name("celeg_matmul_tensor_f16")]]
kernel void celeg_matmul_tensor<half>(
        device half*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        uint2);

template [[host_name("celeg_matmul_tensor_bf16")]]
kernel void celeg_matmul_tensor<bfloat>(
        device bfloat*, device float*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        uint2);
