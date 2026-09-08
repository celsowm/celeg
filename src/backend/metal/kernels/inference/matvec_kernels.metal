/**
 * @brief matvec_kernels — split from vector.metal (full replace, Doxygen only).
 */
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
