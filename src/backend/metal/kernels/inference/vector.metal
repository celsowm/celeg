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
 * @brief One 256-value Q6_K super-block: low nibbles, high bits, scales, scale.
 */
struct CelegQ6kBlock {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

/**
 * @brief Matrix-vector product over Q6_K weights, llama `mul_vec_q6_K` geometry.
 *
 * Two rows per simdgroup and two simdgroups per threadgroup cover four rows;
 * the sixteen `tid` lanes split each 256-value block into 16-value runs and
 * the two `ix` lanes stride the K super-blocks two apart. The four partial
 * sums stay independent for instruction-level parallelism. Launch with 64
 * threads per threadgroup and `(rows + 3) / 4` threadgroups; any other
 * geometry produces wrong rows.
 */
template <typename Input>
void celeg_matvec_q6k_llama_core(device const uchar* weights, Input input, device float* output,
                                 uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                                 uint simd_count, uint group) {
    constexpr uint kRowsPerThread = 2;
    const uint first_row = (group * simd_count + simd) * kRowsPerThread;
    if (first_row >= rows) return;
    device const CelegQ6kBlock* blocks =
        (device const CelegQ6kBlock*)(weights + static_cast<size_t>(first_row) * row_bytes);
    const uint block_count = cols / 256u;

    float sums[kRowsPerThread] = {};
    float lane_inputs[16];
    const uint tid = lane / 2u;
    const uint ix = lane % 2u;
    const uint ip = tid / 8u;
    const uint il = tid % 8u;
    const uint lane_base = 128u * ip + 4u * il;
    const uint ql_base = 64u * ip + 4u * il;
    const uint qh_base = 32u * ip + 4u * il;
    const uint scale_base = 8u * ip + il / 4u;

    for (uint block = ix; block < block_count; block += 2u) {
        const uint block_base = block * 256u;
        for (uint step = 0; step < 4u; ++step) {
            lane_inputs[4u * step] = input.at(block_base + lane_base + step);
            lane_inputs[4u * step + 1u] = input.at(block_base + lane_base + 32u + step);
            lane_inputs[4u * step + 2u] = input.at(block_base + lane_base + 64u + step);
            lane_inputs[4u * step + 3u] = input.at(block_base + lane_base + 96u + step);
        }
        for (uint index = 0; index < kRowsPerThread; ++index) {
            device const CelegQ6kBlock* row_block =
                (device const CelegQ6kBlock*)((device const uchar*)blocks +
                                              static_cast<size_t>(block) * 210u +
                                              static_cast<size_t>(index) * row_bytes);
            float4 partials = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint step = 0; step < 4u; ++step) {
                const uint low = row_block->ql[ql_base + step];
                const uint low2 = row_block->ql[ql_base + 32u + step];
                const uint high = row_block->qh[qh_base + step];
                partials[0] += lane_inputs[4u * step] *
                    (static_cast<int>((low & 0x0fu) | ((high & 0x03u) << 4)) - 32);
                partials[1] += lane_inputs[4u * step + 1u] *
                    (static_cast<int>((low2 & 0x0fu) | ((high & 0x0cu) << 2)) - 32);
                partials[2] += lane_inputs[4u * step + 2u] *
                    (static_cast<int>((low >> 4) | ((high & 0x30u) << 0)) - 32);
                partials[3] += lane_inputs[4u * step + 3u] *
                    (static_cast<int>((low2 >> 4) | ((high & 0xc0u) >> 2)) - 32);
            }
            const float scale = static_cast<float>(row_block->d);
            sums[index] += scale *
                (partials[0] * static_cast<float>(row_block->scales[scale_base]) +
                 partials[1] * static_cast<float>(row_block->scales[scale_base + 2]) +
                 partials[2] * static_cast<float>(row_block->scales[scale_base + 4]) +
                 partials[3] * static_cast<float>(row_block->scales[scale_base + 6]));
        }
    }
    for (uint index = 0; index < kRowsPerThread; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

/**
 * @brief One 256-value Q4_K super-block: two fp16 scales, twelve scale bytes,
 * then 128 bytes of nibbles.
 */
struct CelegQ4kBlock {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

/**
 * @brief Matrix-vector product over Q4_K weights, two rows per thread.
 *
 * Follows the llama.cpp `mul_vec_q4k` decomposition: 64 threads (two
 * simdgroups) per threadgroup cover four rows, the four `ix` lanes stride the
 * K super-blocks four apart, and the eight `it` lanes split each block into
 * register-cached 32-value tiles. Launch with 64 threads per threadgroup and
 * `(rows + 3) / 4` threadgroups; any other geometry produces wrong rows.
 */
template <typename Input, uint MatvecRows>
void celeg_matvec_q4k_core(device const uchar* weights, Input input, device float* output,
                           uint rows, uint cols, uint row_bytes, uint lane, uint simd,
                           uint simd_count, uint group) {
    const uint first_row = (group * simd_count + simd) * MatvecRows;
    if (first_row >= rows) return;
    const uint ix = lane / 8u;
    const uint it = lane % 8u;
    const uint iq = it / 4u;
    const uint ir = it % 4u;

    device const CelegQ4kBlock* blocks_base =
        (device const CelegQ4kBlock*)(weights + static_cast<size_t>(first_row) * row_bytes);
    const uint block_count = cols / 256u;

    float low_inputs[16];
    float high_inputs[16];
    float sums[MatvecRows] = {};

    uint input_base = ix * 256u + 64u * iq + 8u * ir;
    for (uint block = ix; block < block_count; block += 4u) {
        float low_total = 0.0f;
        float high_total = 0.0f;
        float low_base_total = 0.0f;
        float high_base_total = 0.0f;
        for (uint step = 0; step < 8u; ++step) {
            low_inputs[step] = input.at(input_base + step);
            low_total += low_inputs[step];
            low_inputs[step + 8u] = input.at(input_base + 32u + step);
            low_base_total += low_inputs[step + 8u];
            high_inputs[step] = input.at(input_base + 128u + step);
            high_total += high_inputs[step];
            high_inputs[step + 8u] = input.at(input_base + 160u + step);
            high_base_total += high_inputs[step + 8u];
        }

        device const CelegQ4kBlock* step_blocks = blocks_base + block;
        for (uint index = 0; index < MatvecRows; ++index) {
            device const CelegQ4kBlock* row_block =
                (device const CelegQ4kBlock*)((device const uchar*)step_blocks +
                                              static_cast<size_t>(index) * row_bytes);
            device const uint16_t* scales =
                (device const uint16_t*)row_block->scales + iq;
            device const uint16_t* quants =
                (device const uint16_t*)row_block->qs + 16u * iq + 4u * ir;
            const float scale = static_cast<float>(row_block->d);
            const float minimum = static_cast<float>(row_block->dmin);

            const uint16_t packed_scales[4] = {
                static_cast<uint16_t>(scales[0] & 0x3f3fu),
                static_cast<uint16_t>(scales[2] & 0x3f3fu),
                static_cast<uint16_t>(((scales[4] >> 0) & 0x0f0fu) |
                                       ((scales[0] & 0xc0c0u) >> 2)),
                static_cast<uint16_t>(((scales[4] >> 4) & 0x0f0fu) |
                                       ((scales[2] & 0xc0c0u) >> 2)),
            };
            const thread uint8_t* factors = (thread const uint8_t*)packed_scales;

            device const uint16_t* high_quants = quants + 32u;
            float low_accumulators[4] = {};
            float high_accumulators[4] = {};
            for (uint step = 0; step < 4u; ++step) {
                low_accumulators[0] += low_inputs[2u * step] * (quants[step] & 0x000fu);
                low_accumulators[1] += low_inputs[2u * step + 1u] *
                    ((quants[step] & 0x0f00u) >> 8u);
                low_accumulators[2] += low_inputs[2u * step + 8u] *
                    (quants[step] & 0x00f0u);
                low_accumulators[3] += low_inputs[2u * step + 9u] *
                    ((quants[step] & 0xf000u) >> 8u);
                high_accumulators[0] += high_inputs[2u * step] * (high_quants[step] & 0x000fu);
                high_accumulators[1] += high_inputs[2u * step + 1u] *
                    ((high_quants[step] & 0x0f00u) >> 8u);
                high_accumulators[2] += high_inputs[2u * step + 8u] *
                    (high_quants[step] & 0x00f0u);
                high_accumulators[3] += high_inputs[2u * step + 9u] *
                    ((high_quants[step] & 0xf000u) >> 8u);
            }
            sums[index] += scale *
                    ((low_accumulators[0] + low_accumulators[1]) * factors[0] +
                     (low_accumulators[2] + low_accumulators[3]) * factors[1] / 16.0f +
                     (high_accumulators[0] + high_accumulators[1]) * factors[4] +
                     (high_accumulators[2] + high_accumulators[3]) * factors[5] / 16.0f) -
                minimum * (low_total * factors[2] + low_base_total * factors[3] +
                           high_total * factors[6] + high_base_total * factors[7]);
        }
        input_base += 1024u;
    }
    for (uint index = 0; index < MatvecRows; ++index) {
        const float reduced = simd_sum(sums[index]);
        if (lane == 0 && first_row + index < rows) output[first_row + index] = reduced;
    }
}

/**
 * @brief Q4_K llama `mul_vec` geometry — 64 threads (2 SG) cover 4 rows.
 *
 * Mirrors `kernel_mul_mv_q4_K_f32_impl` (NR0=2, NSG=2, QK_K=256). Each lane
 * covers 16 columns split across four 8-value tiles (low 64 + high 64 + high
 * 128 split). Two rows per simdgroup, four rows per threadgroup. Launch with
 * 64 threads and `(rows+3)/4` groups.
 */
template <typename Input>
void celeg_matvec_q4k_llama_core(device const uchar* weights, Input input,
                                 device float* output, uint rows, uint cols,
                                 uint row_bytes, uint lane, uint simd,
                                 uint simd_count, uint group) {
    constexpr uint kRowsPerTG = 4;
    constexpr uint kRowsPerSG = 2;
    const uint first_row = group * kRowsPerTG + simd * kRowsPerSG;
    if (first_row >= rows) return;
    const uint ix = lane / 8u;
    const uint it = lane % 8u;
    const uint iq = it / 4u;
    const uint ir = it % 4u;
    const uint block_count = cols / 256u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint block = ix; block < block_count; block += 4u) {
        const uint base_col = block * 256u;
        float yl[16];
        float yh[16];
        float sumy0 = 0.0f;
        float sumy1 = 0.0f;
        float sumy2 = 0.0f;
        float sumy3 = 0.0f;
        for (uint s = 0; s < 8u; ++s) {
            const uint c0 = base_col + 64u * iq + 8u * ir + s;
            const uint c1 = base_col + 64u * iq + 8u * ir + 32u + s;
            const uint c2 = base_col + 128u + 64u * iq + 8u * ir + s;
            const uint c3 = base_col + 128u + 64u * iq + 8u * ir + 32u + s;
            yl[s] = input.at(c0);
            sumy0 += yl[s];
            yl[8u + s] = input.at(c1);
            sumy1 += yl[8u + s];
            yh[s] = input.at(c2);
            sumy2 += yh[s];
            yh[8u + s] = input.at(c3);
            sumy3 += yh[8u + s];
        }
        for (uint r = 0; r < kRowsPerSG; ++r) {
            const uint row = first_row + r;
            if (row >= rows) break;
            device const CelegQ4kBlock* blk =
                (device const CelegQ4kBlock*)(weights + static_cast<size_t>(row) * row_bytes +
                                              static_cast<size_t>(block) * 144u);
            device const uint16_t* sc = (device const uint16_t*)blk->scales + iq;
            device const uint16_t* q1 = (device const uint16_t*)blk->qs + 16u * iq + 4u * ir;
            const float d = static_cast<float>(blk->d);
            const float dmin = static_cast<float>(blk->dmin);
            uint16_t sc16[4];
            sc16[0] = sc[0] & 0x3f3f;
            sc16[1] = sc[2] & 0x3f3f;
            sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
            device const uint16_t* q2 = q1 + 32u;
            float acc1[4] = {0};
            float acc2[4] = {0};
            for (uint s = 0; s < 4u; ++s) {
                const uint16_t v1 = q1[s];
                const uint16_t v2 = q2[s];
                acc1[0] += yl[2u * s] * (v1 & 0x000Fu);
                acc1[1] += yl[2u * s + 1u] * ((v1 & 0x0f00u) >> 8u);
                acc1[2] += yl[2u * s + 8u] * ((v1 & 0x00f0u) >> 4u);
                acc1[3] += yl[2u * s + 9u] * ((v1 & 0xf000u) >> 12u);
                acc2[0] += yh[2u * s] * (v2 & 0x000Fu);
                acc2[1] += yh[2u * s + 1u] * ((v2 & 0x0f00u) >> 8u);
                acc2[2] += yh[2u * s + 8u] * ((v2 & 0x00f0u) >> 4u);
                acc2[3] += yh[2u * s + 9u] * ((v2 & 0xf000u) >> 12u);
            }
            const float contrib =
                d *
                    ((acc1[0] + acc1[1]) * sc8[0] + (acc1[2] + acc1[3]) * sc8[1] / 16.0f +
                     (acc2[0] + acc2[1]) * sc8[4] + (acc2[2] + acc2[3]) * sc8[5] / 16.0f) -
                dmin *
                    (sumy0 * sc8[2] + sumy1 * sc8[3] + sumy2 * sc8[6] + sumy3 * sc8[7]);
            if (r == 0) sum0 += contrib;
            else sum1 += contrib;
        }
    }
    if (first_row < rows) {
        float out0 = simd_sum(sum0);
        if (lane == 0) output[first_row] = out0;
    }
    if (first_row + 1 < rows) {
        float out1 = simd_sum(sum1);
        if (lane == 0) output[first_row + 1] = out1;
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
    celeg_matvec_q4k_core<CelegDenseInput, 2>(
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
    celeg_matvec_q4k_core<CelegSwigluInput, 2>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}

kernel void celeg_matvec_q4k_rows8(device const uchar* weights [[buffer(0)]],
                             device const float* input [[buffer(1)]],
                             device float* output [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& cols [[buffer(4)]],
                             constant uint& row_bytes [[buffer(5)]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint simd_count [[simdgroups_per_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_core<CelegDenseInput, 8>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q4k_rows8(device const uchar* weights [[buffer(0)]],
                                     device const float* gate_up [[buffer(1)]],
                                     device float* output [[buffer(2)]],
                                     constant uint& rows [[buffer(3)]],
                                     constant uint& cols [[buffer(4)]],
                                     constant uint& row_bytes [[buffer(5)]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint simd [[simdgroup_index_in_threadgroup]],
                                     uint simd_count [[simdgroups_per_threadgroup]],
                                     uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_core<CelegSwigluInput, 8>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols,
        row_bytes, lane, simd, simd_count, group);
}

kernel void celeg_matvec_q4k_llama(device const uchar* weights [[buffer(0)]],
                                   device const float* input [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& row_bytes [[buffer(5)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint simd_count [[simdgroups_per_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_llama_core<CelegDenseInput>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes, lane, simd,
        simd_count, group);
}

kernel void celeg_swiglu_matvec_q4k_llama(device const uchar* weights [[buffer(0)]],
                                          device const float* gate_up [[buffer(1)]],
                                          device float* output [[buffer(2)]],
                                          constant uint& rows [[buffer(3)]],
                                          constant uint& cols [[buffer(4)]],
                                          constant uint& row_bytes [[buffer(5)]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simd [[simdgroup_index_in_threadgroup]],
                                          uint simd_count [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q4k_llama_core<CelegSwigluInput>(
        weights, CelegSwigluInput{gate_up, cols}, output, rows, cols, row_bytes, lane,
        simd, simd_count, group);
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

kernel void celeg_matvec_q6k_llama(device const uchar* weights [[buffer(0)]],
                                   device const float* input [[buffer(1)]],
                                   device float* output [[buffer(2)]],
                                   constant uint& rows [[buffer(3)]],
                                   constant uint& cols [[buffer(4)]],
                                   constant uint& row_bytes [[buffer(5)]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint simd_count [[simdgroups_per_threadgroup]],
                                   uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q6k_llama_core<CelegDenseInput>(
        weights, CelegDenseInput{input}, output, rows, cols, row_bytes,
        lane, simd, simd_count, group);
}

kernel void celeg_swiglu_matvec_q6k_llama(device const uchar* weights [[buffer(0)]],
                                          device const float* gate_up [[buffer(1)]],
                                          device float* output [[buffer(2)]],
                                          constant uint& rows [[buffer(3)]],
                                          constant uint& cols [[buffer(4)]],
                                          constant uint& row_bytes [[buffer(5)]],
                                          uint lane [[thread_index_in_simdgroup]],
                                          uint simd [[simdgroup_index_in_threadgroup]],
                                          uint simd_count [[simdgroups_per_threadgroup]],
                                          uint group [[threadgroup_position_in_grid]]) {
    celeg_matvec_q6k_llama_core<CelegSwigluInput>(
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
