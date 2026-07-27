__global__ void conv_decode_kernel(const __nv_bfloat16* bcx,
                                   const __nv_bfloat16* weight,
                                   __nv_bfloat16* state,
                                   __nv_bfloat16* y,
                                   int hidden,
                                   int cache_length,
                                   int position_value,
                                   const int32_t* position_pointer,
                                   bool use_pointer) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;

    const int position = resolved_position(position_value, position_pointer, use_pointer);
    const float b = bf16_float(bcx[channel]);
    const float c = bf16_float(bcx[hidden + channel]);
    const float x = bf16_float(bcx[2 * hidden + channel]);
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
    const float conv_bf16 = rounded_bf16_float(conv);
    y[channel] = __float2bfloat16(c * conv_bf16);
}

__global__ void conv_prefill_kernel(const __nv_bfloat16* bcx,
                                    const __nv_bfloat16* weight,
                                    __nv_bfloat16* y,
                                    int rows,
                                    int hidden,
                                    int cache_length) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * hidden;
    if (index >= total) return;
    const int row = static_cast<int>(index / hidden);
    const int channel = static_cast<int>(index % hidden);
    const size_t stride = static_cast<size_t>(3) * hidden;

    float conv = 0.0f;
    const size_t weight_offset = static_cast<size_t>(channel) * cache_length;
    for (int tap = 0; tap < cache_length; ++tap) {
        const int source_row = row - (cache_length - 1 - tap);
        if (source_row < 0) continue;
        const __nv_bfloat16* source = bcx + static_cast<size_t>(source_row) * stride;
        const float bx = rounded_bf16_float(
            bf16_float(source[channel]) * bf16_float(source[2 * hidden + channel]));
        conv += bx * bf16_float(weight[weight_offset + tap]);
    }

    const __nv_bfloat16* current = bcx + static_cast<size_t>(row) * stride;
    const float conv_bf16 = rounded_bf16_float(conv);
    y[index] = __float2bfloat16(bf16_float(current[hidden + channel]) * conv_bf16);
}

__global__ void conv_prefill_state_kernel(const __nv_bfloat16* bcx,
                                          __nv_bfloat16* state,
                                          int rows,
                                          int hidden,
                                          int cache_length) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden) return;
    const size_t stride = static_cast<size_t>(3) * hidden;
    for (int slot = 0; slot < cache_length; ++slot) {
        state[static_cast<size_t>(slot) * hidden + channel] = __float2bfloat16(0.0f);
    }
    const int start = rows > cache_length ? rows - cache_length : 0;
    for (int row = start; row < rows; ++row) {
        const __nv_bfloat16* source = bcx + static_cast<size_t>(row) * stride;
        const float bx = rounded_bf16_float(
            bf16_float(source[channel]) * bf16_float(source[2 * hidden + channel]));
        state[static_cast<size_t>(row % cache_length) * hidden + channel] =
            __float2bfloat16(bx);
    }
}

void launch_conv_decode(const __nv_bfloat16* projected_bcx,
                        const __nv_bfloat16* conv_weight,
                        __nv_bfloat16* state, __nv_bfloat16* y,
                        int hidden, int cache_length, int position,
                        cudaStream_t stream) {
    conv_decode_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, conv_weight, state, y, hidden, cache_length,
        position, nullptr, false);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_decode_device(const __nv_bfloat16* projected_bcx,
                               const __nv_bfloat16* conv_weight,
                               __nv_bfloat16* state, __nv_bfloat16* y,
                               int hidden, int cache_length,
                               const int32_t* position,
                               cudaStream_t stream) {
    conv_decode_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, conv_weight, state, y, hidden, cache_length,
        0, position, true);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_conv_prefill(const __nv_bfloat16* projected_bcx,
                         const __nv_bfloat16* conv_weight,
                         __nv_bfloat16* state, __nv_bfloat16* y,
                         int rows, int hidden, int cache_length,
                         cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * hidden;
    conv_prefill_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        projected_bcx, conv_weight, y, rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
    conv_prefill_state_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        projected_bcx, state, rows, hidden, cache_length);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


