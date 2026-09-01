#include <metal_stdlib>

using namespace metal;

kernel void celeg_extract_attention_query_batch(
    device const float* packed [[buffer(0)]],
    device float* query [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& query_width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * query_width;
    if (index >= count) return;
    const uint row = index / query_width;
    const uint column = index % query_width;
    query[index] = packed[static_cast<size_t>(row) * (2 * query_width) + column];
}

kernel void celeg_attention_output_gate(
    device float* output [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& head_wise [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= width) return;
    const uint gate_index = head_wise != 0 ? index / head_dim : index;
    const float scale = 1.0f / (1.0f + exp(-gate[gate_index]));
    output[index] *= scale;
}

kernel void celeg_attention_output_gate_batch(
    device float* output [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& head_wise [[buffer(6)]],
    constant uint& gate_row_stride [[buffer(7)]],
    constant uint& gate_row_offset [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint column = index % width;
    const uint gate_column = head_wise != 0 ? column / head_dim : column;
    const size_t gate_index = static_cast<size_t>(row) * gate_row_stride +
        gate_row_offset + gate_column;
    const float scale = 1.0f / (1.0f + exp(-gate[gate_index]));
    output[index] *= scale;
}
