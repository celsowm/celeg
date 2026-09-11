inline float celeg_swiglu(float gate, float up) {
    return gate / (1.0f + exp(-gate)) * up;
}

inline float celeg_swiglu_relaxed(float gate, float up) {
    return gate / (1.0f + fast::exp(-gate)) * up;
}

inline float celeg_gated_gelu_tanh(float gate, float up) {
    return 0.5f * gate *
        (1.0f + tanh(0.7978845608028654f * (gate + 0.044715f * gate * gate * gate))) * up;
}

inline float celeg_gated_gelu_tanh_relaxed(float gate, float up) {
    const float inner =
        0.7978845608028654f * (gate + 0.044715f * gate * gate * gate);
    const float approximate_tanh = 1.0f - 2.0f / (fast::exp(2.0f * inner) + 1.0f);
    return 0.5f * gate * (1.0f + approximate_tanh) * up;
}

kernel void celeg_swiglu_batch_2d(device const float* gate_up [[buffer(0)]],
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
        celeg_swiglu(gate, up);
}

kernel void celeg_swiglu_batch_2d_relaxed(device const float* gate_up [[buffer(0)]],
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
        celeg_swiglu_relaxed(gate, up);
}

kernel void celeg_gated_gelu_tanh_batch_2d(device const float* gate_up [[buffer(0)]],
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
        celeg_gated_gelu_tanh(gate, up);
}

kernel void celeg_gated_gelu_tanh_batch_2d_relaxed(device const float* gate_up [[buffer(0)]],
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
        celeg_gated_gelu_tanh_relaxed(gate, up);
}
