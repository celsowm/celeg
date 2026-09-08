/**
 * @brief matvec_dense — split from vector.metal (full replace, Doxygen only).
 */
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
