// Benchmark-only parallel short-convolution prefill candidate.
//
// The production celeg_shortconv_batch_ring kernel assigns one thread to each
// channel and walks every token serially. This candidate preserves the exact
// causal ring semantics while exposing rows * width parallelism:
//   1. materialize the gated convolution input for every token/channel;
//   2. compute every token/channel convolution independently;
//   3. publish only the final ring state after all reads of the initial state.
//
// The three-pass structure intentionally prevents races between early-token
// reads of the incoming ring state and final-state publication.

kernel void celeg_shortconv_batch_gate_parallel(
    device const float* projected [[buffer(0)]],
    device float* gated [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;
    const uint row = index / width;
    const uint channel = index - row * width;
    const size_t projected_base = static_cast<size_t>(row) * 3 * width;
    gated[index] = projected[projected_base + channel] *
                   projected[projected_base + 2 * width + channel];
}

kernel void celeg_shortconv_batch_convolve_parallel(
    device const float* projected [[buffer(0)]],
    device const float* taps [[buffer(1)]],
    device const float* initial_state [[buffer(2)]],
    device const float* gated [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& cache_length [[buffer(7)]],
    constant uint& initial_cursor [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = rows * width;
    if (index >= count) return;

    const uint row = index / width;
    const uint channel = index - row * width;
    float convolution = 0.0f;

    for (uint tap = 0; tap < cache_length; ++tap) {
        const uint lag = cache_length - 1u - tap;
        float source = 0.0f;
        if (row >= lag) {
            const uint source_row = row - lag;
            source = gated[static_cast<size_t>(source_row) * width + channel];
        } else {
            const uint slot = (initial_cursor + row + 1u + tap) % cache_length;
            source = initial_state[static_cast<size_t>(slot) * width + channel];
        }
        convolution += source * taps[static_cast<size_t>(tap) * width + channel];
    }

    const size_t projected_base = static_cast<size_t>(row) * 3 * width;
    output[index] = projected[projected_base + width + channel] * convolution;
}

kernel void celeg_shortconv_batch_publish_state_parallel(
    device const float* gated [[buffer(0)]],
    device float* state [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    constant uint& cache_length [[buffer(4)]],
    constant uint& initial_cursor [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint tail_rows = min(rows, cache_length);
    const uint count = tail_rows * width;
    if (index >= count) return;

    const uint tail_index = index / width;
    const uint channel = index - tail_index * width;
    const uint row = rows - tail_rows + tail_index;
    const uint slot = (initial_cursor + row) % cache_length;
    state[static_cast<size_t>(slot) * width + channel] =
        gated[static_cast<size_t>(row) * width + channel];
}
