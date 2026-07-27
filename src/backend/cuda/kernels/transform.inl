__global__ void w4a16_linear_kernel(const __nv_bfloat16* x,
                                    const uint8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta,
                                    int tile_k) {
    constexpr int warps_per_block = 8;
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_act = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block) return;

    const size_t packed_cols = static_cast<size_t>((k + 1) / 2);
    const uint8_t* row_weight = (output_row < n)
        ? (weight + static_cast<size_t>(output_row) * packed_cols)
        : nullptr;
    const float row_scale = (output_row < n) ? scales[output_row] : 0.0f;

    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;

        float accum = 0.0f;
        for (int base_k = 0; base_k < k; base_k += tile_k) {
            const int chunk = (k - base_k) < tile_k ? (k - base_k) : tile_k;

            {
                const int tid = threadIdx.x;
                const int stride = blockDim.x;
                for (int i = tid; i < chunk; i += stride) {
                    s_act[i] = input[base_k + i];
                }
            }
            __syncthreads();

            if (output_row < n) {
                float sum = 0.0f;
                for (int column = lane; column < chunk; column += 32) {
                    sum += bf16_float(s_act[column]) *
                           static_cast<float>(unpack_int4(row_weight, base_k + column));
                }
                accum += warp_sum(sum);
            }

            __syncthreads();
        }
        if (output_row < n && lane == 0) {
            float value = accum * row_scale;
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

__global__ void w8a16_linear_kernel(const __nv_bfloat16* x,
                                    const int8_t* weight,
                                    const float* scales,
                                    __nv_bfloat16* y,
                                    int m, int n, int k, float beta) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int output_row = blockIdx.x * warps_per_block + warp;
    if (warp >= warps_per_block || output_row >= n) return;

    const int8_t* row_weight = weight + static_cast<size_t>(output_row) * k;
    // gridDim.y is capped by the CUDA y-dimension limit. Striding keeps the
    // batched path valid for contexts larger than 65,535 tokens.
    for (int activation_row = blockIdx.y;
         activation_row < m;
         activation_row += gridDim.y) {
        const __nv_bfloat16* input =
            x + static_cast<size_t>(activation_row) * k;
        float sum = 0.0f;
        for (int column = lane; column < k; column += 32) {
            sum += bf16_float(input[column]) *
                   static_cast<float>(row_weight[column]);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            float value = sum * scales[output_row];
            const size_t output_index =
                static_cast<size_t>(activation_row) * n + output_row;
            if (beta != 0.0f) value += beta * bf16_float(y[output_index]);
            y[output_index] = __float2bfloat16(value);
        }
    }
}

__global__ void rmsnorm_kernel(const __nv_bfloat16* x,
                               const __nv_bfloat16* weight,
                               __nv_bfloat16* out,
                               int width,
                               float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* in = x + static_cast<size_t>(row) * width;
    __nv_bfloat16* dst = out + static_cast<size_t>(row) * width;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float v = bf16_float(in[i]);
        sum += v * v;
    }

    __shared__ float warp_sums[32];
    __shared__ float total;
    sum = block_sum(sum, warp_sums, &total);

    __shared__ float inv;
    if (threadIdx.x == 0) {
        inv = rsqrtf(sum / static_cast<float>(width) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(in[i]) * inv);
        dst[i] = __float2bfloat16(normalized * bf16_float(weight[i]));
    }
}

__global__ void residual_kernel(__nv_bfloat16* x,
                                const __nv_bfloat16* residual,
                                int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        x[i] = __float2bfloat16(bf16_float(x[i]) + bf16_float(residual[i]));
    }
}

__global__ void swiglu_fused_kernel(const __nv_bfloat16* gate_up,
                                    __nv_bfloat16* out,
                                    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float gate = bf16_float(gate_up[i]);
        const float up = bf16_float(gate_up[count + i]);
        const float silu = rounded_bf16_float(gate / (1.0f + expf(-gate)));
        out[i] = __float2bfloat16(silu * up);
    }
}

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

__global__ void head_rmsnorm_kernel(__nv_bfloat16* data,
                                    const __nv_bfloat16* norm_weight,
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

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float normalized = rounded_bf16_float(bf16_float(vector[i]) * inv);
        vector[i] = __float2bfloat16(normalized * bf16_float(norm_weight[i]));
    }
}

__global__ void rope_strict_kernel(__nv_bfloat16* data,
                                   const __nv_bfloat16* rope_cos,
                                   const __nv_bfloat16* rope_sin,
                                   int rows,
                                   int heads,
                                   int head_dim,
                                   int position_value,
                                   const int32_t* position_pointer,
                                   int mode) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    const int position = mode == 2 ? row :
        resolved_position(position_value, position_pointer, mode == 1);
    const int half = head_dim / 2;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]);
        const float b = bf16_float(vector[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float s = bf16_float(sin_row[i]);
        const float ac = rounded_bf16_float(a * c);
        const float bs = rounded_bf16_float(b * s);
        const float bc = rounded_bf16_float(b * c);
        const float as = rounded_bf16_float(a * s);
        vector[i] = __float2bfloat16(ac - bs);
        vector[i + half] = __float2bfloat16(bc + as);
    }
}

__global__ void qk_norm_rope_fast_kernel(__nv_bfloat16* data,
                                         const __nv_bfloat16* norm_weight,
                                         const __nv_bfloat16* rope_cos,
                                         const __nv_bfloat16* rope_sin,
                                         int rows,
                                         int heads,
                                         int head_dim,
                                         int position_value,
                                         const int32_t* position_pointer,
                                         int mode,
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

    const int position = mode == 2 ? row :
        resolved_position(position_value, position_pointer, mode == 1);
    const int half = head_dim / 2;
    const __nv_bfloat16* cos_row = rope_cos + static_cast<size_t>(position) * half;
    const __nv_bfloat16* sin_row = rope_sin + static_cast<size_t>(position) * half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float a = bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]);
        const float b = bf16_float(vector[i + half]) * inv *
                        bf16_float(norm_weight[i + half]);
        const float c = bf16_float(cos_row[i]);
        const float s = bf16_float(sin_row[i]);
        vector[i] = __float2bfloat16(a * c - b * s);
        vector[i + half] = __float2bfloat16(b * c + a * s);
    }
}

int attention_threads(int head_dim) {
    int threads = 32;
    while (threads < head_dim && threads < 1024) threads <<= 1;
    return threads;
}

void launch_qk_common(__nv_bfloat16* q,
                      __nv_bfloat16* k,
                      const __nv_bfloat16* q_norm,
                      const __nv_bfloat16* k_norm,
                      const __nv_bfloat16* rope_cos,
                      const __nv_bfloat16* rope_sin,
                      int rows,
                      int q_heads,
                      int kv_heads,
                      int head_dim,
                      int position_value,
                      const int32_t* position_pointer,
                      int mode,
                      float eps,
                      bool fast,
                      cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        qk_norm_rope_fast_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rope_cos, rope_sin, rows, q_heads, head_dim,
            position_value, position_pointer, mode, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, rows, q_heads, head_dim,
            position_value, position_pointer, mode);
        LFM_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode);
        LFM_KERNEL_DEBUG_SYNC(stream);
    }
}

void launch_w8a16_linear(const __nv_bfloat16* x, const int8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid((n + warps_per_block - 1) / warps_per_block, grid_y);
    w8a16_linear_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        x, weight, scales, y, m, n, k, beta);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_w4a16_linear(const __nv_bfloat16* x, const uint8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    // Use 512-element tiles (1 KB smem). For k <= 512, one chunk; for larger k,
    // multiple chunks sharing the same smem buffer per activation row.
    constexpr int tile_k = 512;
    const unsigned grid_x = static_cast<unsigned>((n + warps_per_block - 1) / warps_per_block);
    const unsigned grid_y = static_cast<unsigned>(m < 65535 ? m : 65535);
    const dim3 grid(grid_x, grid_y);
    const size_t smem_size = static_cast<size_t>(tile_k) * sizeof(__nv_bfloat16);
    w4a16_linear_kernel<<<grid, warps_per_block * 32, smem_size, stream>>>(
        x, weight, scales, y, m, n, k, beta, tile_k);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    __nv_bfloat16* out, int rows, int width, float eps,
                    cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(x, weight, out, width, eps);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                         int count, cudaStream_t stream) {
    residual_kernel<<<(count + 255) / 256, 256, 0, stream>>>(x, residual, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                         int count, cudaStream_t stream) {
    swiglu_fused_kernel<<<(count + 255) / 256, 256, 0, stream>>>(gate_up, out, count);
    LFM_KERNEL_DEBUG_SYNC(stream);
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

void launch_qk_norm_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                const __nv_bfloat16* q_norm,
                                const __nv_bfloat16* k_norm,
                                const __nv_bfloat16* rope_cos,
                                const __nv_bfloat16* rope_sin,
                                int q_heads, int kv_heads, int head_dim,
                                int position, float eps,
                                cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, position, nullptr,
                     0, eps, false, stream);
}

void launch_qk_norm_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                       const __nv_bfloat16* q_norm,
                                       const __nv_bfloat16* k_norm,
                                       const __nv_bfloat16* rope_cos,
                                       const __nv_bfloat16* rope_sin,
                                       int q_heads, int kv_heads, int head_dim,
                                       const int32_t* position, float eps,
                                       cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, 0, position,
                     1, eps, false, stream);
}

void launch_qk_norm_rope_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                              const __nv_bfloat16* q_norm,
                              const __nv_bfloat16* k_norm,
                              const __nv_bfloat16* rope_cos,
                              const __nv_bfloat16* rope_sin,
                              int q_heads, int kv_heads, int head_dim,
                              int position, float eps,
                              cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, position, nullptr,
                     0, eps, true, stream);
}

void launch_qk_norm_rope_fast_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                     const __nv_bfloat16* q_norm,
                                     const __nv_bfloat16* k_norm,
                                     const __nv_bfloat16* rope_cos,
                                     const __nv_bfloat16* rope_sin,
                                     int q_heads, int kv_heads, int head_dim,
                                     const int32_t* position, float eps,
                                     cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     1, q_heads, kv_heads, head_dim, 0, position,
                     1, eps, true, stream);
}

void launch_qk_norm_rope_prefill_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                        const __nv_bfloat16* q_norm,
                                        const __nv_bfloat16* k_norm,
                                        const __nv_bfloat16* rope_cos,
                                        const __nv_bfloat16* rope_sin,
                                        int rows, int q_heads, int kv_heads,
                                        int head_dim, float eps,
                                        cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     rows, q_heads, kv_heads, head_dim, 0, nullptr,
                     2, eps, false, stream);
}

void launch_qk_norm_rope_prefill_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                                      const __nv_bfloat16* q_norm,
                                      const __nv_bfloat16* k_norm,
                                      const __nv_bfloat16* rope_cos,
                                      const __nv_bfloat16* rope_sin,
                                      int rows, int q_heads, int kv_heads,
                                      int head_dim, float eps,
                                      cudaStream_t stream) {
    launch_qk_common(q, k, q_norm, k_norm, rope_cos, rope_sin,
                     rows, q_heads, kv_heads, head_dim, 0, nullptr,
                     2, eps, true, stream);
}
