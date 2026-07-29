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
