/**
 * @brief matvec_q4k — split from vector.metal (full replace).
 */
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
