#include <metal_stdlib>
using namespace metal;

kernel void celeg_residual_rmsnorm_batch_cached_128_fast(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* normed [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant float& multiplier [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    threadgroup float* cached [[threadgroup(0)]],
    uint index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]]) {
    if (token >= rows) return;
    const size_t base = static_cast<size_t>(token) * width;
    float sum = 0.0f;
    for (uint i = index; i < width; i += 128) {
        const float value = input[base + i] * multiplier + residual[base + i];
        cached[i] = value;
        sum += value * value;
    }

    threadgroup float partial[4];
    threadgroup float inverse;
    const float reduced = simd_sum(sum);
    if (lane == 0) partial[simd] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index == 0) {
        float total = 0.0f;
        for (uint group = 0; group < 4; ++group) total += partial[group];
        inverse = rsqrt(total / static_cast<float>(width) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = index; i < width; i += 128) {
        const float value = cached[i];
        output[base + i] = value;
        normed[base + i] = value * inverse * weight[i];
    }
}

kernel void celeg_swiglu_batch_2d_fast(
    device const float* gate_up [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 index [[thread_position_in_grid]]) {
    const uint column = index.x;
    const uint token = index.y;
    if (column >= width || token >= rows) return;
    const size_t base = static_cast<size_t>(token) * width * 2;
    const float gate = gate_up[base + column];
    const float up = gate_up[base + width + column];
    output[static_cast<size_t>(token) * width + column] =
        gate / (1.0f + fast::exp(-gate)) * up;
}
