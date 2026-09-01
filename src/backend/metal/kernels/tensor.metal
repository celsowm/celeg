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

void celeg_tensor_q4k_scale_min(device const uchar* scales, uint index,
                                thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

uint celeg_tensor_q4k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[16 + (sub >> 1) * 32 + within];
    return (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
}

half celeg_tensor_q4k_weight(device const uchar* weights, uint row_bytes,
                             uint row, uint column) {
    const device uchar* block = weights + static_cast<size_t>(row) * row_bytes +
        static_cast<size_t>(column / 256) * 144;
    const uint within = column & 255;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_tensor_q4k_scale_min(block + 4, within >> 5, scale, minimum);
    const ushort d_bits = static_cast<ushort>(block[0]) |
        (static_cast<ushort>(block[1]) << 8);
    const ushort dmin_bits = static_cast<ushort>(block[2]) |
        (static_cast<ushort>(block[3]) << 8);
    const float d = static_cast<float>(as_type<half>(d_bits));
    const float dmin = static_cast<float>(as_type<half>(dmin_bits));
    return static_cast<half>(d * static_cast<float>(scale) *
                             static_cast<float>(celeg_tensor_q4k_value(block, within)) -
                             dmin * static_cast<float>(minimum));
}

kernel void celeg_matmul_tensor_q4k(
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
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(tile_k, tile_rows),
        array<int32_t, 2>({1, tile_k}));

    matmul2d<
        matmul2d_descriptor(tile_tokens, tile_rows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }
    for (int offset = 0; offset < static_cast<int>(cols); offset += tile_k) {
        for (int index = static_cast<int>(thread_index); index < tile_rows * tile_k;
             index += 128) {
            const int row = index / tile_k;
            const int column = index % tile_k;
            const int source_row = row_offset + row;
            const int source_column = offset + column;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_column < static_cast<int>(cols)
                ? celeg_tensor_q4k_weight(weights, row_bytes,
                                           static_cast<uint>(source_row),
                                           static_cast<uint>(source_column))
                : static_cast<half>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(tile_k, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, tile_k}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

void celeg_tensor_q5k_scale_min(device const uchar* scales, uint index,
                                thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

uint celeg_tensor_q5k_value(device const uchar* block, uint column) {
    const uint sub = column >> 5;
    const uint within = column & 31;
    const uchar packed = block[48 + (sub >> 1) * 32 + within];
    const uint low = (sub & 1) != 0 ? packed >> 4 : packed & 0x0f;
    return low | (((block[16 + within] >> sub) & 1) << 4);
}

half celeg_tensor_q5k_weight(device const uchar* weights, uint row_bytes,
                             uint row, uint column) {
    const device uchar* block = weights + static_cast<size_t>(row) * row_bytes +
        static_cast<size_t>(column / 256) * 176;
    const uint within = column & 255;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_tensor_q5k_scale_min(block + 4, within >> 5, scale, minimum);
    const ushort d_bits = static_cast<ushort>(block[0]) |
        (static_cast<ushort>(block[1]) << 8);
    const ushort dmin_bits = static_cast<ushort>(block[2]) |
        (static_cast<ushort>(block[3]) << 8);
    const float d = static_cast<float>(as_type<half>(d_bits));
    const float dmin = static_cast<float>(as_type<half>(dmin_bits));
    return static_cast<half>(d * static_cast<float>(scale) *
                             static_cast<float>(celeg_tensor_q5k_value(block, within)) -
                             dmin * static_cast<float>(minimum));
}

kernel void celeg_matmul_tensor_q5k(
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
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(tile_k, tile_rows),
        array<int32_t, 2>({1, tile_k}));

    matmul2d<
        matmul2d_descriptor(tile_tokens, tile_rows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }
    for (int offset = 0; offset < static_cast<int>(cols); offset += tile_k) {
        for (int index = static_cast<int>(thread_index); index < tile_rows * tile_k;
             index += 128) {
            const int row = index / tile_k;
            const int column = index % tile_k;
            const int source_row = row_offset + row;
            const int source_column = offset + column;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_column < static_cast<int>(cols)
                ? celeg_tensor_q5k_weight(weights, row_bytes,
                                           static_cast<uint>(source_row),
                                           static_cast<uint>(source_column))
                : static_cast<half>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(tile_k, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, tile_k}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

int celeg_tensor_q6k_value(device const uchar* block, uint column) {
    const uint half_index = column >> 7;
    const uint index = column & 127;
    const uint lane = index & 31;
    const uint group = index >> 5;
    const device uchar* ql = block + half_index * 64;
    const device uchar* qh = block + 128 + half_index * 32;
    if (group == 0) return (ql[lane] & 0x0f) | (((qh[lane] >> 0) & 3) << 4);
    if (group == 1) return (ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4);
    if (group == 2) return (ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4);
    return (ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4);
}

half celeg_tensor_q6k_weight(device const uchar* weights, uint row_bytes,
                             uint row, uint column) {
    const device uchar* block = weights + static_cast<size_t>(row) * row_bytes +
        static_cast<size_t>(column / 256) * 210;
    const uint within = column & 255;
    const ushort d_bits = static_cast<ushort>(block[208]) |
        (static_cast<ushort>(block[209]) << 8);
    const float d = static_cast<float>(as_type<half>(d_bits));
    const float scale = d * static_cast<float>(
        static_cast<char>(block[192 + within / 16]));
    return static_cast<half>(scale * static_cast<float>(
        celeg_tensor_q6k_value(block, within) - 32));
}

kernel void celeg_matmul_tensor_q6k(
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
    auto output_tile = output_tensor.slice(row_offset, token_offset);
    auto weights_type = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
        weights_tile, dextents<int32_t, 2>(tile_k, tile_rows),
        array<int32_t, 2>({1, tile_k}));

    matmul2d<
        matmul2d_descriptor(tile_tokens, tile_rows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }
    for (int offset = 0; offset < static_cast<int>(cols); offset += tile_k) {
        for (int index = static_cast<int>(thread_index); index < tile_rows * tile_k;
             index += 128) {
            const int row = index / tile_k;
            const int column = index % tile_k;
            const int source_row = row_offset + row;
            const int source_column = offset + column;
            weights_tile[index] = source_row < static_cast<int>(output_rows) &&
                    source_column < static_cast<int>(cols)
                ? celeg_tensor_q6k_weight(weights, row_bytes,
                                           static_cast<uint>(source_row),
                                           static_cast<uint>(source_column))
                : static_cast<half>(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int extent = min(tile_k, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, tile_k}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}
