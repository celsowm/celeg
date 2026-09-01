#include <metal_stdlib>

using namespace metal;

kernel void celeg_extract_attention_query_batch(
    device const float* packed [[buffer(0)]],
    device float* query [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& query_width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * query_width;
    if (index >= count) return;
    const uint row = index / query_width;
    const uint column = index % query_width;
    const uint head = column / head_dim;
    const uint dimension = column % head_dim;
    const size_t source = static_cast<size_t>(row) * (2 * query_width) +
        static_cast<size_t>(head) * (2 * head_dim) + dimension;
    query[index] = packed[source];
}

kernel void celeg_attention_output_gate(
    device float* output [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& head_wise [[buffer(4)]],
    constant uint& packed [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    size_t gate_index = index;
    if (packed != 0) {
        const uint head = index / head_dim;
        const uint dimension = index % head_dim;
        gate_index = static_cast<size_t>(head) * (2 * head_dim) +
            head_dim + dimension;
    } else if (head_wise != 0) {
        gate_index = index / head_dim;
    }
    const float scale = 1.0f / (1.0f + exp(-gate[gate_index]));
    output[index] *= scale;
}

kernel void celeg_attention_output_gate_batch(
    device float* output [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& head_wise [[buffer(5)]],
    constant uint& packed [[buffer(6)]],
    constant uint& gate_row_stride [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index % width;
    size_t gate_index = static_cast<size_t>(row) * gate_row_stride + column;
    if (packed != 0) {
        const uint head = column / head_dim;
        const uint dimension = column % head_dim;
        gate_index = static_cast<size_t>(row) * gate_row_stride +
            static_cast<size_t>(head) * (2 * head_dim) + head_dim + dimension;
    } else if (head_wise != 0) {
        gate_index = static_cast<size_t>(row) * gate_row_stride + column / head_dim;
    }
    const float scale = 1.0f / (1.0f + exp(-gate[gate_index]));
    output[index] *= scale;
}
