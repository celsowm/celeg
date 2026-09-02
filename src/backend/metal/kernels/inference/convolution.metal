kernel void celeg_shortconv_ring(
    device const float* projected [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& cache_length [[buffer(5)]],
    constant uint& cursor [[buffer(6)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= width) return;

    const float value = projected[2 * width + channel] * projected[channel];
    state[static_cast<size_t>(cursor) * width + channel] = value;

    float convolution = 0.0f;
    uint slot = cursor + 1;
    if (slot == cache_length) slot = 0;
    for (uint tap = 0; tap < cache_length; ++tap) {
        convolution += state[static_cast<size_t>(slot) * width + channel] *
                       taps[static_cast<size_t>(tap) * width + channel];
        ++slot;
        if (slot == cache_length) slot = 0;
    }
    output[channel] = projected[width + channel] * convolution;
}

kernel void celeg_shortconv_batch_ring(
    device const float* projected [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& cache_length [[buffer(6)]],
    constant uint& initial_cursor [[buffer(7)]],
    uint channel [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint actual_channel = group * 256 + channel;
    if (actual_channel >= width) return;

    uint cursor = initial_cursor;
    for (uint row = 0; row < rows; ++row) {
        const size_t projected_base = static_cast<size_t>(row) * 3 * width;
        const size_t state_index = static_cast<size_t>(cursor) * width + actual_channel;
        state[state_index] = projected[projected_base + actual_channel] *
            projected[projected_base + 2 * width + actual_channel];

        float convolution = 0.0f;
        uint slot = cursor + 1;
        if (slot == cache_length) slot = 0;
        for (uint tap = 0; tap < cache_length; ++tap) {
            convolution += state[static_cast<size_t>(slot) * width + actual_channel] *
                taps[static_cast<size_t>(tap) * width + actual_channel];
            ++slot;
            if (slot == cache_length) slot = 0;
        }
        output[static_cast<size_t>(row) * width + actual_channel] =
            projected[projected_base + width + actual_channel] * convolution;

        ++cursor;
        if (cursor == cache_length) cursor = 0;
    }
}
