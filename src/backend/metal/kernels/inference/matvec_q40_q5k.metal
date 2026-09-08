/**
 * @brief matvec_q40_q5k — split from vector.metal (full replace, Doxygen only).
 */
/**
 * @brief Per-lane slice of one 256-value Q5_K super-block.
 *
 * Q5_K packs its low nibbles exactly like Q4_K, so a lane owns the same four
 * consecutive `qs` bytes and the same two four-column runs. The fifth bit of
 * each value comes from four consecutive `qh` bytes shared by both runs, which
 * differ only in which bit index they select.
 */
struct CelegQ5kLane {
    uint qs_offset;
    uint qh_offset;
    uint low_column;
    uint high_column;
    uint low_sub;
    uint high_sub;
};

CelegQ5kLane celeg_q5k_lane(uint lane) {
    const uint qs_byte = lane * 4u;
    const uint pair = qs_byte >> 5;
    const uint slot = qs_byte & 31u;
    CelegQ5kLane result;
    result.qs_offset = 48u + qs_byte;
    result.qh_offset = 16u + slot;
    result.low_column = pair * 64u + slot;
    result.high_column = result.low_column + 32u;
    result.low_sub = pair * 2u;
    result.high_sub = result.low_sub + 1u;
    return result;
}

template <typename Input, uint MatvecRows>
void celeg_matvec_q5k_core(device const uchar* weights, Input input, device float* output,
                           uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                           uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * MatvecRows;
    if (first_row >= rows) return;
    const CelegQ5kLane slice = celeg_q5k_lane(lane);

    float sums[MatvecRows] = {};
    const uint blocks = cols / 256u;
    for (uint block = 0; block < blocks; ++block) {
        const uint base = block * 256u;
        float low_input[4];
        float high_input[4];
        float low_total = 0.0f;
        float high_total = 0.0f;
        for (uint step = 0; step < 4u; ++step) {
            low_input[step] = input.at(base + slice.low_column + step);
            high_input[step] = input.at(base + slice.high_column + step);
            low_total += low_input[step];
            high_total += high_input[step];
        }
        for (uint index = 0; index < MatvecRows; ++index) {
            const uint row = first_row + index;
            if (row >= rows) break;
            const device uchar* data = weights + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block) * 176u;
            const float d = celeg_half_to_float(static_cast<ushort>(data[0]) |
                                                (static_cast<ushort>(data[1]) << 8));
            const float dmin = celeg_half_to_float(static_cast<ushort>(data[2]) |
                                                   (static_cast<ushort>(data[3]) << 8));
            uchar low_scale = 0;
            uchar low_minimum = 0;
            uchar high_scale = 0;
            uchar high_minimum = 0;
            celeg_q5k_scale_min(data + 4, slice.low_sub, low_scale, low_minimum);
            celeg_q5k_scale_min(data + 4, slice.high_sub, high_scale, high_minimum);
            device const uchar* qs = data + slice.qs_offset;
            device const uchar* qh = data + slice.qh_offset;
            float low_sum = 0.0f;
            float high_sum = 0.0f;
            for (uint step = 0; step < 4u; ++step) {
                const uint packed = qs[step];
                const uint high_bits = qh[step];
                const uint low_value = (packed & 0x0fu) |
                    (((high_bits >> slice.low_sub) & 1u) << 4);
                const uint high_value = (packed >> 4) |
                    (((high_bits >> slice.high_sub) & 1u) << 4);
                low_sum += static_cast<float>(low_value) * low_input[step];
                high_sum += static_cast<float>(high_value) * high_input[step];
            }
            sums[index] += d * (static_cast<float>(low_scale) * low_sum +
                                static_cast<float>(high_scale) * high_sum) -
                dmin * (static_cast<float>(low_minimum) * low_total +
                        static_cast<float>(high_minimum) * high_total);
        }
    }
    for (uint index = 0; index < MatvecRows; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

/**
 * @brief Per-lane slice of Q4_0 blocks.
 *
 * A Q4_0 block holds 32 values in 16 `qs` bytes, so four lanes cover one block
 * and a simdgroup sweeps eight blocks at a time. A lane owns four consecutive
 * bytes: their low nibbles are four columns and their high nibbles are four
 * more, 16 columns later. Both runs share the one block scale, so eight values
 * cost four byte loads and a single header decode.
 */
struct CelegQ40Lane {
    uint block_offset;
    uint byte_base;
};

CelegQ40Lane celeg_q4_0_lane(uint lane) {
    CelegQ40Lane result;
    result.block_offset = lane >> 2;
    result.byte_base = (lane & 3u) * 4u;
    return result;
}

template <typename Input>
void celeg_matvec_q4_0_core(device const uchar* weights, Input input, device float* output,
                            uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                            uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * kCelegMatvecRows;
    if (first_row >= rows) return;
    const CelegQ40Lane slice = celeg_q4_0_lane(lane);
    const uint total_blocks = cols / 32u;

    float sums[kCelegMatvecRows] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint sweep = 0; sweep * 8u < total_blocks; ++sweep) {
        const uint block = sweep * 8u + slice.block_offset;
        if (block >= total_blocks) break;
        const uint base = block * 32u + slice.byte_base;
        float low_input[4];
        float high_input[4];
        for (uint step = 0; step < 4u; ++step) {
            low_input[step] = input.at(base + step);
            high_input[step] = input.at(base + 16u + step);
        }
        for (uint index = 0; index < kCelegMatvecRows; ++index) {
            const uint row = first_row + index;
            if (row >= rows) break;
            const device uchar* data = weights + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block) * 18u;
            const float d = celeg_half_to_float(static_cast<ushort>(data[0]) |
                                                (static_cast<ushort>(data[1]) << 8));
            device const uchar* qs = data + 2u + slice.byte_base;
            float sum = 0.0f;
            for (uint step = 0; step < 4u; ++step) {
                const uint packed = qs[step];
                sum += (static_cast<float>(packed & 0x0fu) - 8.0f) * low_input[step];
                sum += (static_cast<float>(packed >> 4) - 8.0f) * high_input[step];
            }
            sums[index] += d * sum;
        }
    }
    for (uint index = 0; index < kCelegMatvecRows; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

kernel void celeg_matvec_q4_0(device const uchar* weights [[buffer(0)]],
                              device const float* input [[buffer(1)]],
                              device float* output [[buffer(2)]],
                              constant uint& rows [[buffer(3)]],
                              constant uint& cols [[buffer(4)]],
                              constant uint& row_bytes [[buffer(5)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint simd_count [[simdgroups_per_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4_0_core(weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
                           lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q4_0(
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
    celeg_matvec_q4_0_core(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_matvec_q5k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint simd_count [[simdgroups_per_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q5k_core<CelegDenseInput, 4>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q5k(device const uchar* weights [[buffer(0)]],
                                    device const float* gate_up [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    constant uint& cols [[buffer(4)]],
                                    constant uint& row_bytes [[buffer(5)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint simd_count [[simdgroups_per_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q5k_core<CelegSwigluInput, 4>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}
