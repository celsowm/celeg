__global__ void conv_decode_batch_ptrs_kernel(
    const __nv_bfloat16* bcx,
    const __nv_bfloat16* weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    int rows,
    int hidden,
    int cache_length) {
    const int row = blockIdx.y;
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || channel >= hidden) return;
    const __nv_bfloat16* source = bcx +
        static_cast<size_t>(row) * 3 * hidden;
    __nv_bfloat16* state = states[row];
    const int position = positions[row];
    const float b = bf16_float(source[channel]);
    const float c = bf16_float(source[hidden + channel]);
    const float x = bf16_float(source[2 * hidden + channel]);
    const int cursor = position % cache_length;
    state[static_cast<size_t>(cursor) * hidden + channel] =
        __float2bfloat16(b * x);
    float conv = 0.0f;
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int tap = 0; tap < cache_length; ++tap) {
        const int slot = (cursor + 1 + tap) % cache_length;
        conv += bf16_float(state[static_cast<size_t>(slot) * hidden + channel]) *
                bf16_float(weight[weight_offset + tap]);
    }
    y[static_cast<size_t>(row) * hidden + channel] =
        __float2bfloat16(c * rounded_bf16_float(conv));
}

// A request owns one ShortConv ring.  Tokens belonging to that request must
// therefore be advanced in order, while independent requests/channels remain
// parallel.  `span_offsets` point into the flattened activation rows.
__global__ void conv_ragged_prefill_kernel(
    const __nv_bfloat16* bcx, const __nv_bfloat16* weight,
    __nv_bfloat16* const* states, __nv_bfloat16* y,
    const int32_t* positions, const int32_t* span_offsets,
    const int32_t* span_counts, int hidden, int cache_length) {
    const int request = blockIdx.y;
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;
    const int offset = span_offsets[request];
    const int count = span_counts[request];
    __nv_bfloat16* state = states[request];
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int index = 0; index < count; ++index) {
        const int row = offset + index;
        const __nv_bfloat16* source = bcx + static_cast<size_t>(row) * 3 * hidden;
        const int cursor = positions[row] % cache_length;
        const float b = bf16_float(source[channel]);
        const float c = bf16_float(source[hidden + channel]);
        const float x = bf16_float(source[2 * hidden + channel]);
        state[static_cast<size_t>(cursor) * hidden + channel] =
            __float2bfloat16(b * x);
        float conv = 0.0f;
        for (int tap = 0; tap < cache_length; ++tap) {
            const int slot = (cursor + 1 + tap) % cache_length;
            conv += bf16_float(state[static_cast<size_t>(slot) * hidden + channel]) *
                bf16_float(weight[weight_offset + tap]);
        }
        y[static_cast<size_t>(row) * hidden + channel] =
            __float2bfloat16(c * rounded_bf16_float(conv));
    }
}

void launch_conv_decode_batch_ptrs(
    const __nv_bfloat16* projected_bcx,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    int rows,
    int hidden,
    int cache_length,
    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((hidden + 255) / 256),
                    static_cast<unsigned>(rows));
    conv_decode_batch_ptrs_kernel<<<grid, 256, 0, stream>>>(
        projected_bcx, conv_weight, states, y, positions,
        rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_ragged_prefill(
    const __nv_bfloat16* projected_bcx, const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* states, __nv_bfloat16* y,
    const int32_t* positions, const int32_t* span_offsets,
    const int32_t* span_counts, int requests, int hidden, int cache_length,
    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((hidden + 255) / 256),
                    static_cast<unsigned>(requests));
    conv_ragged_prefill_kernel<<<grid, 256, 0, stream>>>(
        projected_bcx, conv_weight, states, y, positions, span_offsets,
        span_counts, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


