// Benchmark-only F16/BF16 decode matvec A/B.
//
// The scalar kernels mirror the current production dense matvec exactly:
// two output rows per threadgroup, four simdgroups splitting K, scalar loads.
// vec4 preserves that geometry and changes only the inner load width to match
// llama.cpp's dense mul_mv _4 family. vec8 tests two adjacent vectors per lane.

#include <metal_stdlib>

using namespace metal;

inline float celeg_decode_matvec_scalar(half value) {
    return static_cast<float>(value);
}

inline float celeg_decode_matvec_scalar(ushort value) {
    return as_type<float>(static_cast<uint>(value) << 16);
}

inline float4 celeg_decode_matvec_weights(half4 values) {
    return float4(values);
}

inline float4 celeg_decode_matvec_weights(ushort4 values) {
    return as_type<float4>(uint4(values) << 16);
}

template <typename T>
inline void celeg_dense_matvec_scalar_core(
        device const T* weights,
        device const float* input,
        device float* output,
        uint rows,
        uint cols,
        threadgroup float* partial,
        uint lane,
        uint simd,
        uint group) {
    const uint first_row = group * 2u;
    if (first_row >= rows) return;
    float sums[2] = {0.0f, 0.0f};
    for (uint column = simd * 32u + lane; column < cols; column += 128u) {
        const float activation = input[column];
        sums[0] += celeg_decode_matvec_scalar(
            weights[static_cast<size_t>(first_row) * cols + column]) * activation;
        if (first_row + 1u < rows) {
            sums[1] += celeg_decode_matvec_scalar(
                weights[static_cast<size_t>(first_row + 1u) * cols + column]) * activation;
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
        for (uint other = 1; other < 4u; ++other) total += partial[other * 2u + lane];
        output[first_row + lane] = total;
    }
}

template <typename Packed, uint VectorsPerLane>
inline void celeg_dense_matvec_vector_core(
        device const Packed* weights,
        device const float4* input,
        device float* output,
        uint rows,
        uint cols,
        threadgroup float* partial,
        uint lane,
        uint simd,
        uint group) {
    const uint first_row = group * 2u;
    if (first_row >= rows) return;

    const uint values_per_lane = VectorsPerLane * 4u;
    if ((cols % values_per_lane) != 0u) return;
    const uint vectors_per_row = cols / 4u;
    const uint chunks = vectors_per_row / VectorsPerLane;
    float sums[2] = {0.0f, 0.0f};

    for (uint chunk = simd * 32u + lane; chunk < chunks; chunk += 128u) {
        const uint vector_base = chunk * VectorsPerLane;
        #pragma unroll
        for (uint vector_index = 0; vector_index < VectorsPerLane; ++vector_index) {
            const uint vector = vector_base + vector_index;
            const float4 activation = input[vector];
            const size_t row0 = static_cast<size_t>(first_row) * vectors_per_row + vector;
            sums[0] += dot(celeg_decode_matvec_weights(weights[row0]), activation);
            if (first_row + 1u < rows) {
                const size_t row1 = static_cast<size_t>(first_row + 1u) * vectors_per_row + vector;
                sums[1] += dot(celeg_decode_matvec_weights(weights[row1]), activation);
            }
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
        for (uint other = 1; other < 4u; ++other) total += partial[other * 2u + lane];
        output[first_row + lane] = total;
    }
}

kernel void celeg_matvec_f16_scalar_bench(
        device const half* weights [[buffer(0)]],
        device const float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_scalar_core<half>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}

kernel void celeg_matvec_bf16_scalar_bench(
        device const ushort* weights [[buffer(0)]],
        device const float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_scalar_core<ushort>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}

kernel void celeg_matvec_f16_vec4(
        device const half4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_vector_core<half4, 1>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}

kernel void celeg_matvec_bf16_vec4(
        device const ushort4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_vector_core<ushort4, 1>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}

kernel void celeg_matvec_f16_vec8(
        device const half4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_vector_core<half4, 2>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}

kernel void celeg_matvec_bf16_vec8(
        device const ushort4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_dense_matvec_vector_core<ushort4, 2>(
        weights, input, output, rows, cols, partial, lane, simd, group);
}
