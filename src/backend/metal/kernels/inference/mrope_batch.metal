#include <metal_stdlib>

using namespace metal;

inline void celeg_apply_mrope_batch_head(
    device float* values,
    size_t base,
    uint head_dim,
    device const int* position,
    uint section0,
    uint section1,
    uint section2,
    uint interleaved,
    float theta,
    float scale) {
    const uint pairs = section0 + section1 + section2;
    if (pairs == 0u || pairs * 2u > head_dim) return;
    const float rotary_dim = static_cast<float>(pairs * 2u);
    for (uint pair = 0; pair < pairs; ++pair) {
        const uint axis = celeg_mrope_axis_for_pair(
            pair, section0, section1, interleaved);
        const float frequency = pow(
            theta,
            -2.0f * static_cast<float>(pair) / rotary_dim);
        const float angle = static_cast<float>(position[axis]) * frequency;
        const float c = cos(angle);
        const float s = sin(angle);
        const size_t first = base + pair;
        const size_t second = base + pairs + pair;
        const float x = values[first];
        const float y = values[second];
        values[first] = x * c - y * s;
        values[second] = y * c + x * s;
    }
    if (scale != 1.0f) {
        for (uint d = 0; d < head_dim; ++d) values[base + d] *= scale;
    }
}

kernel void celeg_qk_mrope_position_batch(
    device float* query [[buffer(0)]],
    device float* key [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& query_heads [[buffer(3)]],
    constant uint& key_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    device const int* rope_positions [[buffer(6)]],
    constant uint* sections [[buffer(7)]],
    constant float& theta [[buffer(8)]],
    constant float& query_scale [[buffer(9)]],
    constant uint& interleaved [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;

    const uint section_sum = sections[0] + sections[1] + sections[2];
    if (section_sum == 0u || section_sum * 2u > head_dim) return;
    const device int* position = rope_positions + static_cast<size_t>(token) * 3;

    if (head < query_heads) {
        const size_t base = (static_cast<size_t>(token) * query_heads + head) * head_dim;
        celeg_apply_mrope_batch_head(
            query, base, head_dim, position,
            sections[0], sections[1], sections[2], interleaved, theta, query_scale);
    }

    if (head < key_heads) {
        const size_t base = (static_cast<size_t>(token) * key_heads + head) * head_dim;
        celeg_apply_mrope_batch_head(
            key, base, head_dim, position,
            sections[0], sections[1], sections[2], interleaved, theta, 1.0f);
    }
}
