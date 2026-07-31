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
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_swiglu_interleaved(const __nv_bfloat16* gate_up,
                               __nv_bfloat16* out,
                               int rows,
                               int intermediate,
                               cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * intermediate;
    swiglu_interleaved_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        gate_up, out, rows, intermediate);
    CELEG_KERNEL_DEBUG_SYNC(stream);
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
        CELEG_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, positions, rows,
            kv_heads, head_dim, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, positions, rows, q_heads, head_dim);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        rope_batch_positions_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, positions, rows, kv_heads, head_dim);
        CELEG_KERNEL_DEBUG_SYNC(stream);
    }
}



