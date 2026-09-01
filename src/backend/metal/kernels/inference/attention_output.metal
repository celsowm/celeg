#include <metal_stdlib>

using namespace metal;

kernel void celeg_attention_orthogonalize_current_value(
    device float* output [[buffer(0)]],
    device const float* current_value [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& query_heads [[buffer(3)]],
    constant uint& key_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& minimum_norm_squared [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint total_heads = rows * query_heads;
    if (index >= total_heads) return;

    const uint row = index / query_heads;
    const uint query_head = index % query_heads;
    const uint query_heads_per_value = query_heads / key_heads;
    const uint value_head = query_head / query_heads_per_value;
    const size_t output_base =
        (static_cast<size_t>(row) * query_heads + query_head) * head_dim;
    const size_t value_base =
        (static_cast<size_t>(row) * key_heads + value_head) * head_dim;

    float norm_squared = 0.0f;
    float projection = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        const float value = current_value[value_base + d];
        norm_squared += value * value;
        projection += output[output_base + d] * value;
    }
    norm_squared = max(norm_squared, minimum_norm_squared);
    const float coefficient = projection / norm_squared;
    for (uint d = 0; d < head_dim; ++d) {
        output[output_base + d] -= coefficient * current_value[value_base + d];
    }
}
