#include <metal_stdlib>

using namespace metal;

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
    uint index [[thread_position_in_grid]]) {
    const uint head_count = max(query_heads, key_heads);
    const uint token = index / head_count;
    const uint head = index % head_count;
    if (token >= rows) return;

    const uint pairs = head_dim / 2;
    if (sections[0] + sections[1] + sections[2] != pairs) return;
    const device int* position = rope_positions + static_cast<size_t>(token) * 3;

    if (head < query_heads) {
        const size_t base = (static_cast<size_t>(token) * query_heads + head) * head_dim;
        for (uint pair = 0; pair < pairs; ++pair) {
            const uint axis = pair % 3;
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position[axis]) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = query[first];
            const float y = query[second];
            query[first] = (x * c - y * s) * query_scale;
            query[second] = (y * c + x * s) * query_scale;
        }
    }

    if (head < key_heads) {
        const size_t base = (static_cast<size_t>(token) * key_heads + head) * head_dim;
        for (uint pair = 0; pair < pairs; ++pair) {
            const uint axis = pair % 3;
            const float frequency = pow(theta, -2.0f * static_cast<float>(pair) /
                                              static_cast<float>(head_dim));
            const float angle = static_cast<float>(position[axis]) * frequency;
            const float c = cos(angle);
            const float s = sin(angle);
            const size_t first = base + pair;
            const size_t second = base + pairs + pair;
            const float x = key[first];
            const float y = key[second];
            key[first] = x * c - y * s;
            key[second] = y * c + x * s;
        }
    }
}
