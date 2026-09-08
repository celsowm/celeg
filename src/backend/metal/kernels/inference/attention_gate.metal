#include <metal_stdlib>

using namespace metal;

inline size_t celeg_attention_gate_index(
    uint row,
    uint column,
    uint head_dim,
    uint head_wise,
    uint packed,
    uint gate_row_stride) {
    const size_t row_base = static_cast<size_t>(row) * gate_row_stride;
    if (packed != 0) {
        const uint head = column / head_dim;
        const uint dimension = column % head_dim;
        return row_base + static_cast<size_t>(head) * (2 * head_dim) +
            head_dim + dimension;
    }
    if (head_wise != 0) {
        return row_base + column / head_dim;
    }
    return row_base + column;
}

inline float celeg_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

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
    const size_t gate_index = celeg_attention_gate_index(
        0, index, head_dim, head_wise, packed, 0);
    output[index] *= celeg_sigmoid(gate[gate_index]);
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
    const size_t gate_index = celeg_attention_gate_index(
        row, column, head_dim, head_wise, packed, gate_row_stride);
    output[index] *= celeg_sigmoid(gate[gate_index]);
}
