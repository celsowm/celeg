// Dense F16/BF16 decode matvec kernels that reuse one activation load across
// several output rows while preserving Celeg's scalar accumulation order.
//
// Four simdgroups split K exactly as in celeg_matvec_f16/bf16. The only change
// is the number of output rows owned by one threadgroup, so results remain
// bit-identical to the two-row baseline.

#include <metal_stdlib>

using namespace metal;

inline float celeg_dense_rows_element(half value) {
    return static_cast<float>(value);
}

inline float celeg_dense_rows_element(ushort value) {
    return celeg_bf16_to_float(value);
}

template <typename T, uint RowsPerThreadgroup>
inline void celeg_dense_matvec_rows_core(
        device const T* weights,
        device const float* input,
        device float* output,
        uint rows,
        uint cols,
        threadgroup float* partial,
        uint lane,
        uint simd,
        uint group) {
    const uint first_row = group * RowsPerThreadgroup;
    if (first_row >= rows) return;

    float sums[RowsPerThreadgroup] = {};
    for (uint column = simd * 32u + lane; column < cols; column += 128u) {
        const float activation = input[column];
        #pragma unroll
        for (uint row_index = 0; row_index < RowsPerThreadgroup; ++row_index) {
            const uint row = first_row + row_index;
            if (row >= rows) break;
            sums[row_index] += celeg_dense_rows_element(
                weights[static_cast<size_t>(row) * cols + column]) * activation;
        }
    }

    float reduced[RowsPerThreadgroup];
    #pragma unroll
    for (uint row_index = 0; row_index < RowsPerThreadgroup; ++row_index) {
        reduced[row_index] = simd_sum(sums[row_index]);
    }

    if (lane == 0) {
        #pragma unroll
        for (uint row_index = 0; row_index < RowsPerThreadgroup; ++row_index) {
            partial[simd * RowsPerThreadgroup + row_index] = reduced[row_index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd == 0 && lane < RowsPerThreadgroup && first_row + lane < rows) {
        float total = partial[lane];
        #pragma unroll
        for (uint other = 1; other < 4u; ++other) {
            total += partial[other * RowsPerThreadgroup + lane];
        }
        output[first_row + lane] = total;
    }
}

#define CELEG_DENSE_ROWS_KERNEL(NAME, TYPE, ROWS) \
kernel void NAME( \
        device const TYPE* weights [[buffer(0)]], \
        device const float* input [[buffer(1)]], \
        device float* output [[buffer(2)]], \
        constant uint& rows [[buffer(3)]], \
        constant uint& cols [[buffer(4)]], \
        threadgroup float* partial [[threadgroup(0)]], \
        uint lane [[thread_index_in_simdgroup]], \
        uint simd [[simdgroup_index_in_threadgroup]], \
        uint group [[threadgroup_position_in_grid]]) { \
    celeg_dense_matvec_rows_core<TYPE, ROWS>( \
        weights, input, output, rows, cols, partial, lane, simd, group); \
}

CELEG_DENSE_ROWS_KERNEL(celeg_matvec_f16_rows4, half, 4)
CELEG_DENSE_ROWS_KERNEL(celeg_matvec_f16_rows8, half, 8)
CELEG_DENSE_ROWS_KERNEL(celeg_matvec_bf16_rows4, ushort, 4)
CELEG_DENSE_ROWS_KERNEL(celeg_matvec_bf16_rows8, ushort, 8)

#undef CELEG_DENSE_ROWS_KERNEL
