// Benchmark-only F16/BF16 decode matvec row-reuse A/B.
//
// All candidates preserve Celeg's current K partitioning and reduction order:
// four simdgroups split K, scalar products accumulate in the same per-row order,
// and the four simdgroup partials are combined in the same order. The only
// variable is how many output rows share each threadgroup (2, 4, or 8).

#include <metal_stdlib>

using namespace metal;

inline float celeg_rows_element(half value) {
    return static_cast<float>(value);
}

inline float celeg_rows_element(ushort value) {
    return as_type<float>(static_cast<uint>(value) << 16);
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
            sums[row_index] += celeg_rows_element(
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

#define CELEG_ROWS_KERNEL(NAME, TYPE, ROWS) \
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

CELEG_ROWS_KERNEL(celeg_matvec_f16_rows2_bench, half, 2)
CELEG_ROWS_KERNEL(celeg_matvec_f16_rows4_bench, half, 4)
CELEG_ROWS_KERNEL(celeg_matvec_f16_rows8_bench, half, 8)
CELEG_ROWS_KERNEL(celeg_matvec_bf16_rows2_bench, ushort, 2)
CELEG_ROWS_KERNEL(celeg_matvec_bf16_rows4_bench, ushort, 4)
CELEG_ROWS_KERNEL(celeg_matvec_bf16_rows8_bench, ushort, 8)

#undef CELEG_ROWS_KERNEL
