#include <metal_stdlib>

using namespace metal;



kernel void celeg_embedding_f16(device const half* table [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = static_cast<float>(
        table[static_cast<size_t>(token) * width + index]);
}



kernel void celeg_embedding_bf16(device const ushort* table [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index < width) output[index] = celeg_bf16_to_float(
        table[static_cast<size_t>(token) * width + index]);
}

kernel void celeg_matvec(device const float* weights [[buffer(0)]],
                         device const float* input [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 8 + simd;
    if (row >= rows) return;
    float sum = 0.0f;
    const size_t base = static_cast<size_t>(row) * cols;
    for (uint col = lane; col < cols; col += 32) sum += weights[base + col] * input[col];
    const float reduced = simd_sum(sum);
    if (lane == 0) output[row] = reduced;
}

float celeg_matvec_element(half value) {
    return static_cast<float>(value);
}

float celeg_matvec_element(ushort value) {
    return celeg_bf16_to_float(value);
}

template <typename T>
kernel void celeg_matvec_half(device const T* weights [[buffer(0)]],
                               device const float* input [[buffer(1)]],
                               device float* output [[buffer(2)]],
                               constant uint& rows [[buffer(3)]],
                               constant uint& cols [[buffer(4)]],
                               threadgroup float* partial [[threadgroup(0)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]]) {
    const uint row = group * 2;
    if (row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32 + lane; column < cols; column += 128) {
        sums[0] += celeg_matvec_element(weights[static_cast<size_t>(row) * cols + column]) *
            input[column];
        if (row + 1 < rows) {
            sums[1] += celeg_matvec_element(
                weights[static_cast<size_t>(row + 1) * cols + column]) * input[column];
        }
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    if (lane == 0) {
        partial[simd * 2] = reduced0;
        partial[simd * 2 + 1] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 2) {
        float total = partial[lane];
        for (uint other = 1; other < 4; ++other) total += partial[other * 2 + lane];
        if (row + lane < rows) output[row + lane] = total;
    }
}

template [[host_name("celeg_matvec_f16")]]
kernel void celeg_matvec_half<half>(
        device const half*, device const float*, device float*,
        constant uint&, constant uint&, threadgroup float*, uint, uint, uint);

template [[host_name("celeg_matvec_bf16")]]
kernel void celeg_matvec_half<ushort>(
        device const ushort*, device const float*, device float*,
        constant uint&, constant uint&, threadgroup float*, uint, uint, uint);



/**
 * @brief Output rows each simdgroup accumulates in one matvec pass.
 *
 * Holding several rows keeps the activation values in registers across them, so
 * the activation loads, the SwiGLU evaluation and the index arithmetic are paid
 * once per group of rows instead of once per row.
 */
constant uint kCelegMatvecRows = 4;

/// @brief Reads the matrix-vector input straight from a dense float vector.
struct CelegDenseInput {
    device const float* values;

    float at(uint column) const { return values[column]; }
};

/**
 * @brief Applies SwiGLU to a packed gate/up pair and feeds the result in.
 *
 * Fusing the activation into the down projection removes both the separate
 * elementwise dispatch and the round trip of the intermediate vector through
 * device memory.
 */
struct CelegSwigluInput {
    device const float* gate_up;
    uint cols;

    float at(uint column) const {
        const float gate = gate_up[column];
        return gate / (1.0f + exp(-gate)) * gate_up[cols + column];
    }
};

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
 * @brief Per-lane slice of one 256-value Q6_K super-block.
 *
 * A lane owns four consecutive `ql` bytes, which carry eight values: the four
 * low nibbles form one run of four columns and the four high nibbles form a
 * second run 64 columns later. Both runs are four-column aligned, so each sits
 * entirely inside one 16-value scale group. The two runs share the same four
 * `qh` bytes, differing only in which bit pair supplies the high two bits, so
 * eight values cost four `ql` bytes, four `qh` bytes and two scale bytes rather
 * than the two block-header decodes per value the scalar form paid.
 */
struct CelegQ6kLane {
    uint ql_offset;
    uint qh_offset;
    uint low_column;
    uint high_column;
    uint low_shift;
    uint high_shift;
};

CelegQ6kLane celeg_q6k_lane(uint lane) {
    const uint ql_byte = lane * 4u;
    const uint half_index = ql_byte >> 6;
    const uint within_half = ql_byte & 63u;
    const uint upper = within_half >> 5;
    const uint slot = within_half & 31u;
    CelegQ6kLane result;
    result.ql_offset = half_index * 64u + within_half;
    result.qh_offset = 128u + half_index * 32u + slot;
    result.low_column = half_index * 128u + upper * 32u + slot;
    result.high_column = result.low_column + 64u;
    result.low_shift = upper * 2u;
    result.high_shift = result.low_shift + 4u;
    return result;
}

template <typename Input, uint MatvecRows>
void celeg_matvec_q6k_core(device const uchar* weights, Input input, device float* output,
                           uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                           uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * MatvecRows;
    if (first_row >= rows) return;
    const CelegQ6kLane slice = celeg_q6k_lane(lane);
    const uint low_scale = slice.low_column >> 4;
    const uint high_scale = slice.high_column >> 4;

    float sums[MatvecRows] = {};
    const uint blocks = cols / 256u;
    for (uint block = 0; block < blocks; ++block) {
        const uint base = block * 256u;
        float low_input[4];
        float high_input[4];
        for (uint step = 0; step < 4u; ++step) {
            low_input[step] = input.at(base + slice.low_column + step);
            high_input[step] = input.at(base + slice.high_column + step);
        }
        for (uint index = 0; index < MatvecRows; ++index) {
            const uint row = first_row + index;
            if (row >= rows) break;
            const device uchar* data = weights + static_cast<size_t>(row) * row_bytes +
                static_cast<size_t>(block) * 210u;
            const float d = celeg_half_to_float(static_cast<ushort>(data[208]) |
                                                (static_cast<ushort>(data[209]) << 8));
            const float low_factor = d * static_cast<float>(
                static_cast<char>(data[192 + low_scale]));
            const float high_factor = d * static_cast<float>(
                static_cast<char>(data[192 + high_scale]));
            device const uchar* ql = data + slice.ql_offset;
            device const uchar* qh = data + slice.qh_offset;
            float low_sum = 0.0f;
            float high_sum = 0.0f;
            for (uint step = 0; step < 4u; ++step) {
                const uint packed = ql[step];
                const uint high_bits = qh[step];
                const int low_value = static_cast<int>(
                    (packed & 0x0fu) | (((high_bits >> slice.low_shift) & 3u) << 4)) - 32;
                const int high_value = static_cast<int>(
                    (packed >> 4) | (((high_bits >> slice.high_shift) & 3u) << 4)) - 32;
                low_sum += static_cast<float>(low_value) * low_input[step];
                high_sum += static_cast<float>(high_value) * high_input[step];
            }
            sums[index] += low_factor * low_sum + high_factor * high_sum;
        }
    }
    for (uint index = 0; index < MatvecRows; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

/**
 * @brief Per-lane slice of one 256-value Q4_K super-block.
 *
 * A lane owns four consecutive `qs` bytes, whose low nibbles are a run of four
 * columns and whose high nibbles are a second run 32 columns later. Each run is
 * four-column aligned, so it lies entirely inside one 32-value sub-block and
 * needs a single scale/minimum unpack per row instead of one per value.
 */
struct CelegQ4kLane {
    uint qs_offset;
    uint low_column;
    uint high_column;
    uint low_sub;
    uint high_sub;
};

CelegQ4kLane celeg_q4k_lane(uint lane) {
    const uint qs_byte = lane * 4u;
    const uint pair = qs_byte >> 5;
    const uint slot = qs_byte & 31u;
    CelegQ4kLane result;
    result.qs_offset = 16u + qs_byte;
    result.low_column = pair * 64u + slot;
    result.high_column = result.low_column + 32u;
    result.low_sub = pair * 2u;
    result.high_sub = result.low_sub + 1u;
    return result;
}

/**
 * @brief Matrix-vector product over Q4_K weights, four rows per simdgroup.
 *
 * The `dmin` correction is folded into the sum of the activations covering each
 * sub-block, so it costs one multiply per sub-block instead of the two extra
 * fused multiply-adds per value the per-element form paid.
 */
template <typename Input, uint MatvecRows>
void celeg_matvec_q4k_core(device const uchar* weights, Input input, device float* output,
                           uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                           uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * MatvecRows;
    if (first_row >= rows) return;
    const CelegQ4kLane slice = celeg_q4k_lane(lane);

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
                static_cast<size_t>(block) * 144u;
            const float d = celeg_half_to_float(static_cast<ushort>(data[0]) |
                                                (static_cast<ushort>(data[1]) << 8));
            const float dmin = celeg_half_to_float(static_cast<ushort>(data[2]) |
                                                   (static_cast<ushort>(data[3]) << 8));
            uchar low_scale = 0;
            uchar low_minimum = 0;
            uchar high_scale = 0;
            uchar high_minimum = 0;
            celeg_q4k_scale_min(data + 4, slice.low_sub, low_scale, low_minimum);
            celeg_q4k_scale_min(data + 4, slice.high_sub, high_scale, high_minimum);
            device const uchar* qs = data + slice.qs_offset;
            float low_sum = 0.0f;
            float high_sum = 0.0f;
            for (uint step = 0; step < 4u; ++step) {
                const uint packed = qs[step];
                low_sum += static_cast<float>(packed & 0x0fu) * low_input[step];
                high_sum += static_cast<float>(packed >> 4) * high_input[step];
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

kernel void celeg_matvec_q4k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint simd_count [[simdgroups_per_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_core<CelegDenseInput, 4>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q4k(device const uchar* weights [[buffer(0)]],
                                    device const float* gate_up [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    constant uint& cols [[buffer(4)]],
                                    constant uint& row_bytes [[buffer(5)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint simd_count [[simdgroups_per_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_core<CelegSwigluInput, 4>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}

kernel void celeg_matvec_q6k(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint simd_count [[simdgroups_per_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q6k_core<CelegDenseInput, 4>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q6k(device const uchar* weights [[buffer(0)]],
                                    device const float* gate_up [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant uint& rows [[buffer(3)]],
                                    constant uint& cols [[buffer(4)]],
                                    constant uint& row_bytes [[buffer(5)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint simd_count [[simdgroups_per_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q6k_core<CelegSwigluInput, 4>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}

#define CELEG_MATVEC_ROWS8_KERNEL(NAME, CORE) \
kernel void NAME(device const uchar* weights [[buffer(0)]], \
                 device const float* input [[buffer(1)]], \
                 device float* output [[buffer(2)]], \
                 constant uint& rows [[buffer(3)]], \
                 constant uint& cols [[buffer(4)]], \
                 constant uint& row_bytes [[buffer(5)]], \
                 uint lane [[thread_index_in_simdgroup]], \
                 uint simd [[simdgroup_index_in_threadgroup]], \
                 uint simd_count [[simdgroups_per_threadgroup]], \
                 uint group [[threadgroup_position_in_grid]]) { \
    CORE<CelegDenseInput, 8>(weights, CelegDenseInput{input}, output, rows, cols, \
                             row_bytes, lane, simd, simd_count, group); \
}

CELEG_MATVEC_ROWS8_KERNEL(celeg_matvec_q4k_rows8, celeg_matvec_q4k_core)
CELEG_MATVEC_ROWS8_KERNEL(celeg_matvec_q5k_rows8, celeg_matvec_q5k_core)
CELEG_MATVEC_ROWS8_KERNEL(celeg_matvec_q6k_rows8, celeg_matvec_q6k_core)

#undef CELEG_MATVEC_ROWS8_KERNEL

kernel void celeg_swiglu_matvec_q6k_rows8(
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
    celeg_matvec_q6k_core<CelegSwigluInput, 8>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}


kernel void celeg_embedding_q4k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 144 + block_index * 144;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    const uint sub = column >> 5;
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q4k_scale_min(block + 4, sub, scale, minimum);
    output[index] = d * static_cast<float>(scale) *
                    static_cast<float>(celeg_q4k_value(block, column)) -
                    dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q6k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) *
                                (width / 256) * 210 + block_index * 210;
    const float d = celeg_half_to_float(static_cast<ushort>(block[208]) |
                                        (static_cast<ushort>(block[209]) << 8));
    const uint sub = column / 16;
    const float scale = d * static_cast<float>(static_cast<char>(block[192 + sub]));
    output[index] = scale * static_cast<float>(celeg_q6k_value(block, column) - 32);
}

kernel void celeg_embedding_q4_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 18 +
        static_cast<size_t>(index / 32) * 18;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * celeg_q4_0_value(block, index);
}

kernel void celeg_embedding_q5k(device const uchar* weights [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint& width [[buffer(2)]],
                                constant uint& token [[buffer(3)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint block_index = index / 256;
    const uint column = index & 255;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 256) * 176 +
        static_cast<size_t>(block_index) * 176;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    const float dmin = celeg_half_to_float(static_cast<ushort>(block[2]) |
                                           (static_cast<ushort>(block[3]) << 8));
    uchar scale = 0;
    uchar minimum = 0;
    celeg_q5k_scale_min(block + 4, column >> 5, scale, minimum);
    output[index] = d * static_cast<float>(scale) * celeg_q5k_value(block, column) -
        dmin * static_cast<float>(minimum);
}

kernel void celeg_embedding_q8_0(device const uchar* weights [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& width [[buffer(2)]],
                                 constant uint& token [[buffer(3)]],
                                 uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const device uchar* block = weights + static_cast<size_t>(token) * (width / 32) * 34 +
        static_cast<size_t>(index / 32) * 34;
    const float d = celeg_half_to_float(static_cast<ushort>(block[0]) |
                                        (static_cast<ushort>(block[1]) << 8));
    output[index] = d * static_cast<float>(static_cast<char>(block[2 + (index & 31)]));
}
