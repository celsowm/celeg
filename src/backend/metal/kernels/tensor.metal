#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

/**
 * @brief Weight-tile geometry shared by every batched matmul in this file.
 *
 * The staging step decodes whole 32-value quantization sub-blocks, so a
 * @c kTileK of 64 gives each of the 128 threads exactly one sub-block of the
 * 64x64 tile and halves the barrier count relative to a 32-column tile.
 */
constant int kCelegTileRows = 64;
constant int kCelegTileTokens = 128;
constant int kCelegTileK = 64;
constant int kCelegTileThreads = 128;
constant int kCelegBlockValues = 32;

template <typename T>
kernel void celeg_matmul_tensor(
        device const T* weights [[buffer(0)]],
        device half* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        threadgroup T* weights_tile [[threadgroup(0)]],
        uint thread_index [[thread_index_in_threadgroup]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    const int row_offset = static_cast<int>(grid.x) * kCelegTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegTileTokens;
    const int row_extent = min(kCelegTileRows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegTileTokens, static_cast<int>(rows) - token_offset);
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
        weights_tile, dextents<int32_t, 2>(kCelegTileK, kCelegTileRows),
        array<int32_t, 2>({1, kCelegTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegTileTokens, kCelegTileRows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tile), decltype(weights_type), float>();
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

template [[host_name("celeg_matmul_tensor_f16")]]
kernel void celeg_matmul_tensor<half>(
        device const half*, device half*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup half*, uint, uint2);

template [[host_name("celeg_matmul_tensor_bf16")]]
kernel void celeg_matmul_tensor<bfloat>(
        device const bfloat*, device half*, device float*,
        constant uint&, constant uint&, constant uint&, constant uint&,
        threadgroup bfloat*, uint, uint2);

void celeg_tensor_scale_min(device const uchar* scales, uint index,
                            thread uchar& scale, thread uchar& minimum) {
    if (index < 4) {
        scale = scales[index] & 63;
        minimum = scales[index + 4] & 63;
        return;
    }
    scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
}

/**
 * @brief Decodes one 32-value Q4_0 block into the weight tile.
 *
 * Every decoder in this family writes a whole sub-block from a single block
 * header, so the scale is unpacked once per 32 values rather than once per
 * value as the per-element interface required.
 */
struct CelegTensorQ4_0 {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 18;
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        for (uint index = 0; index < 16u; ++index) {
            const uint packed = block[2 + index];
            destination[index] =
                static_cast<half>(d * (static_cast<float>(packed & 0x0fu) - 8.0f));
            destination[index + 16u] =
                static_cast<half>(d * (static_cast<float>(packed >> 4) - 8.0f));
        }
    }
};

/// @brief Decodes one 32-value Q8_0 block into the weight tile.
struct CelegTensorQ8_0 {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 32) * 34;
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        for (uint index = 0; index < 32u; ++index) {
            destination[index] = static_cast<half>(
                d * static_cast<float>(static_cast<char>(block[2 + index])));
        }
    }
};

/// @brief Decodes one 32-value Q4_K sub-block into the weight tile.
struct CelegTensorQ4K {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 256) * 144;
        const uint within = column & 255u;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_tensor_scale_min(block + 4, sub, scale, minimum);
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        const ushort dmin_bits = static_cast<ushort>(
            static_cast<uint>(block[2]) | (static_cast<uint>(block[3]) << 8));
        const float dmin = static_cast<float>(as_type<half>(dmin_bits));
        const float factor = d * static_cast<float>(scale);
        const float bias = dmin * static_cast<float>(minimum);
        const device uchar* qs = block + 16 + (sub >> 1) * 32;
        const bool high = (sub & 1u) != 0;
        for (uint index = 0; index < 32u; ++index) {
            const uint packed = qs[index];
            const uint value = high ? (packed >> 4) : (packed & 0x0fu);
            destination[index] = static_cast<half>(factor * static_cast<float>(value) - bias);
        }
    }
};

/// @brief Decodes one 32-value Q5_K sub-block into the weight tile.
struct CelegTensorQ5K {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 256) * 176;
        const uint within = column & 255u;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_tensor_scale_min(block + 4, sub, scale, minimum);
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        const ushort dmin_bits = static_cast<ushort>(
            static_cast<uint>(block[2]) | (static_cast<uint>(block[3]) << 8));
        const float dmin = static_cast<float>(as_type<half>(dmin_bits));
        const float factor = d * static_cast<float>(scale);
        const float bias = dmin * static_cast<float>(minimum);
        const device uchar* qs = block + 48 + (sub >> 1) * 32;
        const device uchar* qh = block + 16;
        const bool high = (sub & 1u) != 0;
        for (uint index = 0; index < 32u; ++index) {
            const uint packed = qs[index];
            const uint low = high ? (packed >> 4) : (packed & 0x0fu);
            const uint value = low | (((qh[index] >> sub) & 1u) << 4);
            destination[index] = static_cast<half>(factor * static_cast<float>(value) - bias);
        }
    }
};

/// @brief Decodes one 32-value Q6_K run into the weight tile.
struct CelegTensorQ6K {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 256) * 210;
        const uint within = column & 255u;
        const uint half_index = within >> 7;
        const uint group = (within & 127u) >> 5;
        const device uchar* ql = block + half_index * 64u + (group & 1u) * 32u;
        const device uchar* qh = block + 128u + half_index * 32u;
        const uint shift = group * 2u;
        const bool high = group >= 2u;
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[208]) | (static_cast<uint>(block[209]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        const float first = d * static_cast<float>(
            static_cast<char>(block[192 + (within >> 4)]));
        const float second = d * static_cast<float>(
            static_cast<char>(block[192 + (within >> 4) + 1]));
        for (uint index = 0; index < 32u; ++index) {
            const uint packed = ql[index];
            const uint low = high ? (packed >> 4) : (packed & 0x0fu);
            const int value = static_cast<int>(
                low | (((qh[index] >> shift) & 3u) << 4)) - 32;
            destination[index] = static_cast<half>(
                (index < 16u ? first : second) * static_cast<float>(value));
        }
    }
};

/**
 * @brief Batched matmul over natively quantized weights.
 *
 * The weight tile is staged as `half` one quantization sub-block per thread,
 * then handed to the same cooperative `matmul2d` the dense kernels use.
 */
template <typename Decoder>
void celeg_matmul_tensor_quantized(
        device const uchar* weights, device half* input, device float* output,
        uint rows, uint cols, uint output_rows, uint output_stride, uint row_bytes,
        Decoder decoder, threadgroup half* weights_tile, uint thread_index, uint2 grid) {
    const int row_offset = static_cast<int>(grid.x) * kCelegTileRows;
    const int token_offset = static_cast<int>(grid.y) * kCelegTileTokens;
    const int row_extent = min(kCelegTileRows, static_cast<int>(output_rows) - row_offset);
    const int token_extent = min(kCelegTileTokens, static_cast<int>(rows) - token_offset);
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
        weights_tile, dextents<int32_t, 2>(kCelegTileK, kCelegTileRows),
        array<int32_t, 2>({1, kCelegTileK}));

    matmul2d<
        matmul2d_descriptor(kCelegTileTokens, kCelegTileRows, dynamic_extent,
                            false, true, false,
                            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> operation;
    auto result = operation.get_destination_cooperative_tensor<
        decltype(input_tensor), decltype(weights_type), float>();
    for (uint16_t index = 0; index < result.get_capacity(); ++index) {
        if (result.is_valid_element(index)) result[index] = 0.0f;
    }

    constexpr int blocks_per_row = kCelegTileK / kCelegBlockValues;
    const int tile_row = static_cast<int>(thread_index) / blocks_per_row;
    const int tile_block = static_cast<int>(thread_index) % blocks_per_row;
    const int source_row = row_offset + tile_row;
    threadgroup half* destination =
        weights_tile + tile_row * kCelegTileK + tile_block * kCelegBlockValues;

    for (int offset = 0; offset < static_cast<int>(cols); offset += kCelegTileK) {
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
        const int extent = min(kCelegTileK, static_cast<int>(cols) - offset);
        auto input_slice = tensor(
            input + static_cast<size_t>(token_offset) * cols + offset,
            dextents<int32_t, 2>(extent, token_extent),
            array<int32_t, 2>({1, static_cast<int32_t>(cols)}));
        auto weights_slice = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(
            weights_tile, dextents<int32_t, 2>(extent, row_extent),
            array<int32_t, 2>({1, kCelegTileK}));
        operation.run(input_slice, weights_slice, result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    result.store(output_tile);
}

#define CELEG_TENSOR_QUANTIZED_MATMUL(NAME, DECODER) \
kernel void NAME( \
        device const uchar* weights [[buffer(0)]], \
        device half* input [[buffer(1)]], \
        device float* output [[buffer(2)]], \
        constant uint& rows [[buffer(3)]], \
        constant uint& cols [[buffer(4)]], \
        constant uint& output_rows [[buffer(5)]], \
        constant uint& output_stride [[buffer(6)]], \
        constant uint& row_bytes [[buffer(7)]], \
        threadgroup half* weights_tile [[threadgroup(0)]], \
        uint thread_index [[thread_index_in_threadgroup]], \
        uint2 grid [[threadgroup_position_in_grid]]) { \
    celeg_matmul_tensor_quantized(weights, input, output, rows, cols, output_rows, \
                                  output_stride, row_bytes, DECODER{}, weights_tile, \
                                  thread_index, grid); \
}

CELEG_TENSOR_QUANTIZED_MATMUL(celeg_matmul_tensor_q4_0, CelegTensorQ4_0)
CELEG_TENSOR_QUANTIZED_MATMUL(celeg_matmul_tensor_q8_0, CelegTensorQ8_0)
CELEG_TENSOR_QUANTIZED_MATMUL(celeg_matmul_tensor_q4k, CelegTensorQ4K)
CELEG_TENSOR_QUANTIZED_MATMUL(celeg_matmul_tensor_q5k, CelegTensorQ5K)
CELEG_TENSOR_QUANTIZED_MATMUL(celeg_matmul_tensor_q6k, CelegTensorQ6K)

#undef CELEG_TENSOR_QUANTIZED_MATMUL
