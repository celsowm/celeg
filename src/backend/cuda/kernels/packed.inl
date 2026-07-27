__global__ void split_qkv_rows_kernel(const __nv_bfloat16* qkv,
                                      __nv_bfloat16* q,
                                      __nv_bfloat16* k,
                                      __nv_bfloat16* v,
                                      int rows,
                                      int q_width,
                                      int kv_width) {
    const size_t total = static_cast<size_t>(rows) *
                         static_cast<size_t>(q_width + 2 * kv_width);
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int width = q_width + 2 * kv_width;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    if (column < q_width) {
        q[static_cast<size_t>(row) * q_width + column] = qkv[index];
    } else if (column < q_width + kv_width) {
        k[static_cast<size_t>(row) * kv_width + column - q_width] = qkv[index];
    } else {
        v[static_cast<size_t>(row) * kv_width + column - q_width - kv_width] = qkv[index];
    }
}

__global__ void swiglu_interleaved_kernel(const __nv_bfloat16* gate_up,
                                          __nv_bfloat16* out,
                                          int rows,
                                          int intermediate) {
    const size_t total = static_cast<size_t>(rows) * intermediate;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / intermediate);
    const int column = static_cast<int>(index % intermediate);
    const __nv_bfloat16* source = gate_up +
        static_cast<size_t>(row) * 2 * intermediate;
    const float gate = bf16_float(source[column]);
    const float up = bf16_float(source[intermediate + column]);
    const float silu = rounded_bf16_float(gate / (1.0f + expf(-gate)));
    out[index] = __float2bfloat16(silu * up);
}

__global__ void rope_batch_positions_kernel(__nv_bfloat16* data,
                                            const __nv_bfloat16* rope_cos,
                                            const __nv_bfloat16* rope_sin,
                                            const int32_t* positions,
                                            int rows,
                                            int heads,
                                            int head_dim) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    const int position = positions[row];
    const int half = head_dim / 2;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]);
        const float b = bf16_float(vector[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float ss = bf16_float(sin_row[i]);
        const float ac = rounded_bf16_float(a * c);
        const float bs = rounded_bf16_float(b * ss);
        const float bc = rounded_bf16_float(b * c);
        const float as = rounded_bf16_float(a * ss);
        vector[i] = __float2bfloat16(ac - bs);
        vector[i + half] = __float2bfloat16(bc + as);
    }
}

__global__ void qk_norm_rope_fast_batch_positions_kernel(
    __nv_bfloat16* data,
    const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions,
    int rows,
    int heads,
    int head_dim,
    float eps) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float value = bf16_float(vector[i]);
        sum += value * value;
    }
    __shared__ float warp_sums[32];
    __shared__ float total;
    sum = block_sum(sum, warp_sums, &total);
    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();
    const int position = positions[row];
    const int half = head_dim / 2;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]);
        const float b = bf16_float(vector[i + half]) * inv *
                        bf16_float(norm_weight[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float ss = bf16_float(sin_row[i]);
        vector[i] = __float2bfloat16(a * c - b * ss);
        vector[i + half] = __float2bfloat16(b * c + a * ss);
    }
}

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

__global__ void scatter_bf16_rows_kernel(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    destinations[row][column] = source[index];
}

__global__ void scatter_decode_state_kernel(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    *sampled_destinations[row] = sampled[row];
    *position_destinations[row] = positions[row] + 1;
}

__global__ void scatter_bf16_selected_rows_kernel(
    const __nv_bfloat16* source, const int32_t* source_rows,
    __nv_bfloat16* const* destinations, int rows, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int request = static_cast<int>(index / width);
    destinations[request][index % width] =
        source[static_cast<size_t>(source_rows[request]) * width + index % width];
}

__global__ void scatter_selected_decode_state_kernel(
    const int32_t* sampled, const int32_t* positions,
    const int32_t* source_rows, int32_t* const* sampled_destinations,
    int32_t* const* position_destinations, int rows) {
    const int request = blockIdx.x * blockDim.x + threadIdx.x;
    if (request >= rows) return;
    const int row = source_rows[request];
    *sampled_destinations[request] = sampled[row];
    *position_destinations[request] = positions[row] + 1;
}

__global__ void increment_position_kernel(int32_t* position) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ++(*position);
}

void launch_split_qkv_rows(const __nv_bfloat16* qkv,
                           __nv_bfloat16* q,
                           __nv_bfloat16* k,
                           __nv_bfloat16* v,
                           int rows,
                           int q_width,
                           int kv_width,
                           cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) *
                         static_cast<size_t>(q_width + 2 * kv_width);
    split_qkv_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        qkv, q, k, v, rows, q_width, kv_width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_interleaved(const __nv_bfloat16* gate_up,
                               __nv_bfloat16* out,
                               int rows,
                               int intermediate,
                               cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * intermediate;
    swiglu_interleaved_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        gate_up, out, rows, intermediate);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_qk_norm_rope_batch_positions(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    const __nv_bfloat16* q_norm,
    const __nv_bfloat16* k_norm,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    float eps,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        qk_norm_rope_fast_batch_positions_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rope_cos, rope_sin, positions, rows,
            q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, positions, rows,
            kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, positions, rows, q_heads, head_dim);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, positions, rows, kv_heads, head_dim);
        LFM_KERNEL_DEBUG_SYNC(stream);
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

void launch_scatter_bf16_rows(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows,
    int width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    scatter_bf16_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        source, destinations, rows, width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_bf16_selected_rows(
    const __nv_bfloat16* source, const int32_t* source_rows,
    __nv_bfloat16* const* destinations, int rows, int width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * width;
    scatter_bf16_selected_rows_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        source, source_rows, destinations, rows, width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_decode_state(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows,
    cudaStream_t stream) {
    scatter_decode_state_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sampled, positions, sampled_destinations,
        position_destinations, rows);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_scatter_selected_decode_state(
    const int32_t* sampled, const int32_t* positions,
    const int32_t* source_rows, int32_t* const* sampled_destinations,
    int32_t* const* position_destinations, int rows, cudaStream_t stream) {
    scatter_selected_decode_state_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        sampled, positions, source_rows, sampled_destinations,
        position_destinations, rows);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_increment_position(int32_t* position, cudaStream_t stream) {
    increment_position_kernel<<<1, 1, 0, stream>>>(position);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

