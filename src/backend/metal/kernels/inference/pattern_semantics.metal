#include <metal_stdlib>

using namespace metal;

bool celeg_attention_causal_visible(int query_position, int key_position) {
    return query_position >= 0 && key_position >= 0 && key_position <= query_position;
}

uint celeg_attention_sliding_first_candidate(uint query_position, uint window_size) {
    if (window_size == 0) return 0;
    const uint sequence_length = query_position + 1;
    return sequence_length > window_size ? sequence_length - window_size : 0;
}

bool celeg_attention_sliding_visible(int query_position, int key_position,
                                     uint window_size) {
    if (!celeg_attention_causal_visible(query_position, key_position)) return false;
    if (window_size == 0) return true;
    return static_cast<uint>(key_position) >=
        celeg_attention_sliding_first_candidate(
            static_cast<uint>(query_position), window_size);
}

kernel void celeg_attention_pattern_semantics_probe(
    device uint* output [[buffer(0)]],
    constant int& query_position [[buffer(1)]],
    constant int& key_position [[buffer(2)]],
    constant uint& window_size [[buffer(3)]]) {
    output[0] = celeg_attention_causal_visible(query_position, key_position) ? 1u : 0u;
    output[1] = celeg_attention_sliding_first_candidate(
        static_cast<uint>(max(query_position, 0)), window_size);
    output[2] = celeg_attention_sliding_visible(
        query_position, key_position, window_size) ? 1u : 0u;
}
