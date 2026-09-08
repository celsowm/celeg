/**
 * @brief matvec_q6k — split from vector.metal (full replace).
 */
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

