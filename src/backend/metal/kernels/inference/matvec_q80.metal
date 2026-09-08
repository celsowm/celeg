/**
 * @brief matvec_q80 — split from vector.metal (full replace, Doxygen only).
 */
/**
 * @brief Per-lane slice of Q8_0 blocks.
 *
 * A Q8_0 block holds 32 signed bytes, so eight lanes cover one block and a
 * simdgroup sweeps four blocks at a time. A lane owns four consecutive bytes
 * and decodes them under one block scale.
 */
struct CelegQ80Lane {
    uint block_offset;
    uint byte_base;
};

CelegQ80Lane celeg_q8_0_lane(uint lane) {
    CelegQ80Lane result;
    result.block_offset = lane >> 3;
    result.byte_base = (lane & 7u) * 4u;
    return result;
}

template <typename Input, uint MatvecRows>
void celeg_matvec_q8_0_core(device const uchar* weights, Input input, device float* output,
                            uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                            uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * MatvecRows;
    if (first_row >= rows) return;
    const CelegQ80Lane slice = celeg_q8_0_lane(lane);
    const uint total_blocks = cols / 32u;

    float sums[MatvecRows] = {};
    for (uint sweep = 0; sweep * 4u < total_blocks; ++sweep) {
        const uint block = sweep * 4u + slice.block_offset;
        if (block >= total_blocks) break;
        const uint base = block * 32u + slice.byte_base;
        float values[4];
        for (uint step = 0; step < 4u; ++step) values[step] = input.at(base + step);
        for (uint index = 0; index < MatvecRows; ++index) {
            const uint row = first_row + index;
            if (row >= rows) break;
            const device uchar* data = weights + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block) * 34u;
            const float d = celeg_half_to_float(static_cast<ushort>(data[0]) |
                                                (static_cast<ushort>(data[1]) << 8));
            device const uchar* qs = data + 2u + slice.byte_base;
            float sum = 0.0f;
            for (uint step = 0; step < 4u; ++step) {
                sum += static_cast<float>(static_cast<char>(qs[step])) * values[step];
            }
            sums[index] += d * sum;
        }
    }
    for (uint index = 0; index < MatvecRows; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

kernel void celeg_matvec_q8_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint simd_count [[simdgroups_per_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q8_0_core<CelegDenseInput, 4>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_matvec_q8_0_rows8(device const uchar* weights [[buffer(0)]],
                                    device const float* input [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    constant uint& cols [[buffer(4)]],
                                    constant uint& row_bytes [[buffer(5)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint simd_count [[simdgroups_per_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q8_0_core<CelegDenseInput, 8>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q8_0_rows8(
        device const uchar* weights [[buffer(0)]],
        device const float* gate_up [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& row_bytes [[buffer(5)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint simd_count [[simdgroups_per_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q8_0_core<CelegSwigluInput, 8>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}

/**
 * @brief Q8_0 matvec geometry matching the pinned llama.cpp MPP baseline.
 *
 * Four SIMD groups split K for the same pair of output rows. Each lane consumes
 * eight adjacent activations, and the threadgroup reduction combines the four
 * partial sums before storing the two results.
 */
template <typename Input>
void celeg_matvec_q8_0_m5_core(
        device const uchar* weights, Input input, device float* output,
        uint rows, uint cols, uint row_bytes, threadgroup float* partial,
        uint lane, uint simd, uint simd_count, uint group) {
    const uint first_row = group * 2u;
    if (first_row >= rows) return;
    const uint block_lane = lane >> 2;
    const uint value_lane = (lane & 3u) * 8u;
    const uint first_block = simd * 8u + block_lane;
    const uint total_blocks = cols / 32u;
    float sums[2] = {0.0f, 0.0f};
    for (uint block = first_block; block < total_blocks;
         block += simd_count * 8u) {
        const uint column = block * 32u + value_lane;
        float activations[8];
        for (uint index = 0; index < 8u; ++index) {
            activations[index] = input.at(column + index);
        }
        for (uint row_index = 0; row_index < 2u; ++row_index) {
            const uint row = first_row + row_index;
            if (row >= rows) break;
            const device uchar* data = weights + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block) * 34u;
            const float scale = celeg_half_to_float(static_cast<ushort>(data[0]) |
                                                    (static_cast<ushort>(data[1]) << 8));
            device const char* quants =
                reinterpret_cast<device const char*>(data + 2u + value_lane);
            float sum = 0.0f;
            for (uint index = 0; index < 8u; ++index) {
                sum += static_cast<float>(quants[index]) * activations[index];
            }
            sums[row_index] += scale * sum;
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2u] = reduced0;
        partial[simd * 2u + 1u] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2u && first_row + lane < rows) {
        float total = partial[lane];
        for (uint group_index = 1; group_index < simd_count; ++group_index) {
            total += partial[group_index * 2u + lane];
        }
        output[first_row + lane] = total;
    }
}

kernel void celeg_matvec_q8_0_m5(
        device const uchar* weights [[buffer(0)]],
        device const float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& row_bytes [[buffer(5)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint simd_count [[simdgroups_per_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q8_0_m5_core(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        partial, lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q8_0_m5(
        device const uchar* weights [[buffer(0)]],
        device const float* gate_up [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& row_bytes [[buffer(5)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint simd_count [[simdgroups_per_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q8_0_m5_core(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols, row_bytes,
        partial, lane, simd, simd_count, group);
}
