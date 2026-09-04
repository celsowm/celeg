// Benchmark-only F16/BF16 decode matvec candidates.
//
// Production dense matvec uses two output rows per threadgroup and four
// simdgroups to split K. These candidates preserve that geometry and reduction
// exactly, changing only the inner load width from scalar values to float4 +
// half4/raw-BF16 vectors. vec4 matches llama.cpp's dense mul_mv _4 family;
// vec8 tests whether two adjacent vectors per lane amortize loop overhead
// further on Apple M5.

#include <metal_stdlib>

using namespace metal;

inline float4 celeg_decode_matvec_weights(half4 values) {
    return float4(values);
}

inline float4 celeg_decode_matvec_weights(ushort4 values) {
    return as_type<float4>(uint4(values) << 16);
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

    // The benchmark shapes are multiples of 8, so vec4/vec8 cover K exactly.
    // Keep a scalar tail here so the candidate remains correct for arbitrary
    // dense shapes if the harness is extended later.
    const uint covered = chunks * VectorsPerLane * 4u;
    for (uint column = covered + simd * 32u + lane;
         column < cols; column += 128u) {
        const float activation = reinterpret_cast<device const float*>(input)[column];
        const size_t row0 = static_cast<size_t>(first_row) * cols + column;
        if constexpr (is_same_v<Packed, half4>) {
            const device half* scalar_weights =
                reinterpret_cast<device const half*>(weights);
            sums[0] += static_cast<float>(scalar_weights[row0]) * activation;
            if (first_row + 1u < rows) {
                sums[1] += static_cast<float>(
                    scalar_weights[static_cast<size_t>(first_row + 1u) * cols + column]) *
                    activation;
            }
        } else {
            const device ushort* scalar_weights =
                reinterpret_cast<device const ushort*>(weights);
            const auto bf16 = [](ushort bits) {
                return as_type<float>(static_cast<uint>(bits) << 16);
            };
            sums[0] += bf16(scalar_weights[row0]) * activation;
            if (first_row + 1u < rows) {
                sums[1] += bf16(
                    scalar_weights[static_cast<size_t>(first_row + 1u) * cols + column]) *
                    activation;
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
        for (uint other = 1; other < 4u; ++other) {
            total += partial[other * 2u + lane];
        }
        output[first_row + lane] = total;
    }
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
