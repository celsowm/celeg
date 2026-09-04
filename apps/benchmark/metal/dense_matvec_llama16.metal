// Benchmark-only F16/BF16 decode matvec candidate matching llama.cpp's
// dense mul_mv _4 geometry for contiguous matrices.
//
// Baseline: Celeg production geometry, 2 rows/TG, 4 simdgroups, scalar loads.
// Candidate: llama.cpp-style NB=32/NF=16 mapping, four float4 dot products per
// lane, same 2 rows/TG and 4 simdgroups. BF16 is represented as ushort4 so the
// benchmark remains portable to runtime MSL environments without bfloat4.

#include <metal_stdlib>

using namespace metal;

inline float celeg_llama16_scalar(half value) {
    return static_cast<float>(value);
}

inline float celeg_llama16_scalar(ushort value) {
    return as_type<float>(static_cast<uint>(value) << 16);
}

inline float4 celeg_llama16_vector(half4 value) {
    return float4(value);
}

inline float4 celeg_llama16_vector(ushort4 value) {
    return as_type<float4>(uint4(value) << 16);
}

template <typename T>
inline void celeg_llama16_scalar_core(
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
        sums[0] += celeg_llama16_scalar(
            weights[static_cast<size_t>(first_row) * cols + column]) * activation;
        if (first_row + 1u < rows) {
            sums[1] += celeg_llama16_scalar(
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
        for (uint other = 1; other < 4u; ++other) {
            total += partial[other * 2u + lane];
        }
        output[first_row + lane] = total;
    }
}

template <typename Packed>
inline void celeg_llama16_vector_core(
        device const Packed* weights4,
        device const float4* input4,
        device float* output,
        uint rows,
        uint cols,
        threadgroup float* shared,
        uint lane,
        uint simd,
        uint group) {
    constexpr uint NSG = 4u;
    constexpr uint NB = 32u;
    constexpr uint NF = 16u;
    constexpr uint NF4 = NF / 4u;

    const uint first_row = group * 2u;
    if (first_row >= rows) return;
    if ((cols % NB) != 0u) return;

    const uint nb = cols / NB;
    const uint vectors_per_row = cols / 4u;
    const uint ix = lane / 2u;
    const uint il = lane % 2u;
    const uint ib0 = simd * NF + ix;

    float sums[2] = {0.0f, 0.0f};
    size_t y_index = static_cast<size_t>(ib0 * NB + il * NF) / 4u;

    for (uint ib = ib0; ib < nb; ib += NSG * NF) {
        float4 activations[NF4];
        #pragma unroll
        for (uint i = 0; i < NF4; ++i) {
            activations[i] = input4[y_index + i];
        }

        #pragma unroll
        for (uint row_index = 0; row_index < 2u; ++row_index) {
            const uint row = first_row + row_index;
            if (row >= rows) break;
            const size_t x_index = static_cast<size_t>(row) * vectors_per_row +
                static_cast<size_t>(ib * NB + il * NF) / 4u;
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < NF4; ++i) {
                sum += dot(celeg_llama16_vector(weights4[x_index + i]), activations[i]);
            }
            sums[row_index] += sum;
        }

        y_index += static_cast<size_t>(NSG * NF * 32u) / 4u;
    }

    // Match llama.cpp helper_mv_reduce_and_write<2>: 32 floats per row.
    threadgroup float* row0_shared = shared;
    threadgroup float* row1_shared = shared + 32u;
    if (simd == 0) {
        row0_shared[lane] = 0.0f;
        row1_shared[lane] = 0.0f;
    }
    const float reduced0 = simd_sum(sums[0]);
    const float reduced1 = simd_sum(sums[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        row0_shared[simd] = reduced0;
        row1_shared[simd] = reduced1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const float total0 = simd_sum(row0_shared[lane]);
        const float total1 = simd_sum(row1_shared[lane]);
        if (lane == 0) output[first_row] = total0;
        if (lane == 1 && first_row + 1u < rows) output[first_row + 1u] = total1;
    }
}

kernel void celeg_matvec_f16_scalar_llama16_bench(
        device const half* weights [[buffer(0)]],
        device const float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_llama16_scalar_core(weights, input, output, rows, cols,
                              partial, lane, simd, group);
}

kernel void celeg_matvec_bf16_scalar_llama16_bench(
        device const ushort* weights [[buffer(0)]],
        device const float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* partial [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_llama16_scalar_core(weights, input, output, rows, cols,
                              partial, lane, simd, group);
}

kernel void celeg_matvec_f16_llama16(
        device const half4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* shared [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_llama16_vector_core(weights, input, output, rows, cols,
                              shared, lane, simd, group);
}

kernel void celeg_matvec_bf16_llama16(
        device const ushort4* weights [[buffer(0)]],
        device const float4* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        threadgroup float* shared [[threadgroup(0)]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    celeg_llama16_vector_core(weights, input, output, rows, cols,
                              shared, lane, simd, group);
}
