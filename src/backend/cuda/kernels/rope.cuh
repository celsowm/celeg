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
        CELEG_KERNEL_DEBUG_SYNC(stream);
        qk_norm_rope_fast_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
    } else {
        head_rmsnorm_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, q_norm, rows, q_heads, head_dim, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        head_rmsnorm_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, eps);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, rope_cos, rope_sin, rows, q_heads, head_dim,
            position_value, position_pointer, mode);
        CELEG_KERNEL_DEBUG_SYNC(stream);
        rope_strict_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, rope_cos, rope_sin, rows, kv_heads, head_dim,
            position_value, position_pointer, mode);
        CELEG_KERNEL_DEBUG_SYNC(stream);
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

void launch_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                        const __nv_bfloat16* rope_cos,
                        const __nv_bfloat16* rope_sin,
                        int q_heads, int kv_heads, int head_dim,
                        int position, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    rope_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, rope_cos, rope_sin, 1, q_heads, head_dim, position, nullptr, 0);
    rope_strict_kernel<<<kv_heads, threads, 0, stream>>>(
        k, rope_cos, rope_sin, 1, kv_heads, head_dim, position, nullptr, 0);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                               const __nv_bfloat16* rope_cos,
                               const __nv_bfloat16* rope_sin,
                               int q_heads, int kv_heads, int head_dim,
                               const int32_t* position, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    rope_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, rope_cos, rope_sin, 1, q_heads, head_dim, 0, position, 1);
    rope_strict_kernel<<<kv_heads, threads, 0, stream>>>(
        k, rope_cos, rope_sin, 1, kv_heads, head_dim, 0, position, 1);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_rope_prefill(__nv_bfloat16* q, __nv_bfloat16* k,
                         const __nv_bfloat16* rope_cos,
                         const __nv_bfloat16* rope_sin,
                         int rows, int q_heads, int kv_heads, int head_dim,
                         cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    rope_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, rope_cos, rope_sin, rows, q_heads, head_dim, 0, nullptr, 2);
    rope_strict_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, rope_cos, rope_sin, rows, kv_heads, head_dim, 0, nullptr, 2);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_rope_batch_positions(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* rope_cos, const __nv_bfloat16* rope_sin,
    const int32_t* positions, int rows, int q_heads, int kv_heads,
    int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    rope_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, rope_cos, rope_sin, rows, q_heads, head_dim, 0, positions, 1);
    rope_strict_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, rope_cos, rope_sin, rows, kv_heads, head_dim, 0, positions, 1);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__device__ __forceinline__ float dynamic_yarn_correction_dimension(
    float rotations, int rotary_dimension, float theta, int original_context) {
    const float numerator = static_cast<float>(rotary_dimension) *
        logf(static_cast<float>(original_context) /
             (rotations * 6.28318530717958647692f));
    const float denominator = 2.0f * logf(theta);
    return numerator / denominator;
}

/// `rotary_dimension` must be the actual rotated width (2 * rotary_pairs),
/// not head_dim -- for partial-rotary checkpoints (rotary_fraction < 1.0)
/// those differ, and using head_dim here silently understates the frequency
/// falloff across the rotated dims. Invisible whenever rotary_fraction == 1.0,
/// which is why this was wrong for a long time before a partial-rotary
/// checkpoint (Qwen3.5) surfaced it -- see
/// docs/QWEN3_5_NVFP4_FP8_SUPPORT_PLAN.md Phase 6.
__device__ __forceinline__ float scaled_rope_frequency(
    float theta, int pair, int rotary_dimension, int position,
    CudaRopeScaling scaling) {
    float base = theta;
    if (scaling.kind == 2 && scaling.original_context > 0 &&
        position > scaling.original_context) {
        const float context = static_cast<float>(position);
        const float ratio = scaling.factor * context /
            static_cast<float>(scaling.original_context) - (scaling.factor - 1.0f);
        const int denominator = rotary_dimension - 2 > 2 ? rotary_dimension - 2 : 2;
        base *= powf(fmaxf(1.0f, ratio),
                     static_cast<float>(rotary_dimension) /
                     static_cast<float>(denominator));
    }
    float frequency = powf(base, -2.0f * static_cast<float>(pair) /
                           static_cast<float>(rotary_dimension));
    if (scaling.kind == 1) {
        frequency /= scaling.factor;
    } else if (scaling.kind == 3) {
        const float low = floorf(dynamic_yarn_correction_dimension(
            scaling.beta_fast, rotary_dimension, theta, scaling.original_context));
        const float high = ceilf(dynamic_yarn_correction_dimension(
            scaling.beta_slow, rotary_dimension, theta, scaling.original_context));
        const float clipped_low = fmaxf(0.0f, low);
        const float clipped_high = fminf(
            static_cast<float>(rotary_dimension - 1), high);
        const float span = fmaxf(0.001f, clipped_high - clipped_low);
        const float ramp = fminf(1.0f, fmaxf(0.0f,
            (static_cast<float>(pair) - clipped_low) / span));
        const float interpolated = frequency / scaling.factor;
        frequency = frequency * (1.0f - ramp) + interpolated * ramp;
    } else if (scaling.kind == 4) {
        if (pair < scaling.factor_count) {
            const float factor = position > scaling.original_context
                ? scaling.long_factors[pair] : scaling.short_factors[pair];
            frequency /= factor;
        }
    } else if (scaling.kind == 5) {
        const float wavelength = 6.28318530717958647692f / frequency;
        if (wavelength > static_cast<float>(scaling.original_context) /
                         scaling.low_frequency_factor) {
            frequency /= scaling.factor;
        } else if (wavelength <= static_cast<float>(scaling.original_context) /
                                  scaling.high_frequency_factor) {
        } else {
            const float span = scaling.low_frequency_factor -
                scaling.high_frequency_factor;
            const float blend = span > 0.0f
                ? (wavelength * scaling.high_frequency_factor /
                   static_cast<float>(scaling.original_context) - 1.0f) / span
                : 0.0f;
            frequency /= 1.0f + fminf(1.0f, fmaxf(0.0f, blend)) *
                (scaling.factor - 1.0f);
        }
    }
    return frequency;
}

__global__ void dynamic_qk_norm_rope_kernel(
    __nv_bfloat16* data, const __nv_bfloat16* norm_weight,
    int rows, int heads, int head_dim, int position_value,
    const int32_t* position_pointer, int mode, float theta,
    int rotary_pairs, float eps, bool normalize, CudaRopeScaling scaling,
    float attention_scale, RopePairingKind pairing) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    __shared__ float warp_sums[32];
    __shared__ float total;
    __shared__ float inv;
    if (normalize) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float value = bf16_float(vector[i]);
            sum += value * value;
        }
        sum = block_sum(sum, warp_sums, &total);
        if (threadIdx.x == 0) inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
        __syncthreads();
    } else if (threadIdx.x == 0) {
        inv = 1.0f;
    }
    __syncthreads();
    const int position = mode == 2 ? row :
        resolved_position(position_value, position_pointer, mode == 1);
    for (int i = threadIdx.x; i < rotary_pairs; i += blockDim.x) {
        /// AdjacentPairs rotates consecutive components (2i, 2i+1), matching
        /// llama.cpp's "normal" RoPE and the row order its conversion writes;
        /// SplitHalf rotates (i, rotary_pairs + i), the NEOX/HuggingFace order.
        const int low = pairing == RopePairingKind::AdjacentPairs ? 2 * i : i;
        const int high = pairing == RopePairingKind::AdjacentPairs
            ? 2 * i + 1 : rotary_pairs + i;
        const float a = bf16_float(vector[low]) * inv *
            (normalize ? bf16_float(norm_weight[low]) : 1.0f);
        const float b = bf16_float(vector[high]) * inv *
            (normalize ? bf16_float(norm_weight[high]) : 1.0f);
        const float frequency = scaled_rope_frequency(
            theta, i, 2 * rotary_pairs, position, scaling);
        const float angle = static_cast<float>(position) * frequency;
        const float c = cosf(angle);
        const float s = sinf(angle);
        vector[low] = __float2bfloat16(a * c - b * s);
        vector[high] = __float2bfloat16(b * c + a * s);
    }
    if (normalize) {
        for (int i = threadIdx.x + 2 * rotary_pairs; i < head_dim; i += blockDim.x) {
            vector[i] = __float2bfloat16(bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]));
        }
    }
    __syncthreads();
    if (attention_scale != 1.0f) {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            vector[i] = __float2bfloat16(bf16_float(vector[i]) * attention_scale);
        }
    }
}

void launch_dynamic_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, int position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling, RopePairingKind pairing, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    const int pairs = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction) / 2;
    const float query_attention_scale = scaling.kind == 3
        ? scaling.attention_factor * scaling.attention_factor
        : 1.0f;
    dynamic_qk_norm_rope_kernel<<<q_heads, threads, 0, stream>>>(
        q, q_norm, 1, q_heads, head_dim, position, nullptr, 0,
        rope_theta, pairs, eps, normalize, scaling, query_attention_scale,
        pairing);
    if (k) {
        dynamic_qk_norm_rope_kernel<<<kv_heads, threads, 0, stream>>>(
            k, k_norm, 1, kv_heads, head_dim, position, nullptr, 0,
            rope_theta, pairs, eps, normalize, scaling, 1.0f, pairing);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_dynamic_qk_norm_rope_device(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, const int32_t* position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling, RopePairingKind pairing, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    const int pairs = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction) / 2;
    const float query_attention_scale = scaling.kind == 3
        ? scaling.attention_factor * scaling.attention_factor
        : 1.0f;
    dynamic_qk_norm_rope_kernel<<<q_heads, threads, 0, stream>>>(
        q, q_norm, 1, q_heads, head_dim, 0, position, 1,
        rope_theta, pairs, eps, normalize, scaling, query_attention_scale,
        pairing);
    if (k) {
        dynamic_qk_norm_rope_kernel<<<kv_heads, threads, 0, stream>>>(
            k, k_norm, 1, kv_heads, head_dim, 0, position, 1,
            rope_theta, pairs, eps, normalize, scaling, 1.0f, pairing);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_dynamic_qk_norm_rope_prefill(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int rows, int q_heads, int kv_heads, int head_dim,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling, RopePairingKind pairing, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    const int pairs = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction) / 2;
    const float query_attention_scale = scaling.kind == 3
        ? scaling.attention_factor * scaling.attention_factor
        : 1.0f;
    dynamic_qk_norm_rope_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, q_norm, rows, q_heads, head_dim, 0, nullptr, 2,
        rope_theta, pairs, eps, normalize, scaling, query_attention_scale,
        pairing);
    if (k) {
        dynamic_qk_norm_rope_kernel<<<rows * kv_heads, threads, 0, stream>>>(
            k, k_norm, rows, kv_heads, head_dim, 0, nullptr, 2,
            rope_theta, pairs, eps, normalize, scaling, 1.0f, pairing);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__device__ __forceinline__ int mrope_axis_for_pair_device(
    int pair, int section0, int section1, bool interleaved) {
    if (interleaved) return pair % 3;
    return pair < section0 ? 0 : (pair < section0 + section1 ? 1 : 2);
}

__global__ void dynamic_mrope_qk_norm_rope_kernel(
    __nv_bfloat16* data, const __nv_bfloat16* norm_weight,
    int rows, int heads, int head_dim, const int32_t* positions,
    int section0, int section1, int section2, bool interleaved,
    float theta, int rotary_pairs, float eps, bool normalize,
    CudaRopeScaling scaling, float attention_scale) {
    const int block = blockIdx.x;
    const int row = block / heads;
    const int head = block % heads;
    if (row >= rows) return;
    __nv_bfloat16* vector = data +
        (static_cast<size_t>(row) * heads + head) * head_dim;
    __shared__ float warp_sums[32];
    __shared__ float total;
    __shared__ float inv;
    if (normalize) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float value = bf16_float(vector[i]);
            sum += value * value;
        }
        sum = block_sum(sum, warp_sums, &total);
        if (threadIdx.x == 0) inv = rsqrtf(sum / static_cast<float>(head_dim) + eps);
        __syncthreads();
    } else if (threadIdx.x == 0) {
        inv = 1.0f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rotary_pairs; i += blockDim.x) {
        const int axis = mrope_axis_for_pair_device(i, section0, section1, interleaved);
        const float position = static_cast<float>(positions[axis]);
        const float frequency = scaled_rope_frequency(
            theta, i, 2 * rotary_pairs, static_cast<int>(position), scaling);
        const float angle = position * frequency;
        const float a = bf16_float(vector[i]) * inv *
            (normalize ? bf16_float(norm_weight[i]) : 1.0f);
        const float b = bf16_float(vector[rotary_pairs + i]) * inv *
            (normalize ? bf16_float(norm_weight[rotary_pairs + i]) : 1.0f);
        const float c = cosf(angle);
        const float s = sinf(angle);
        vector[i] = __float2bfloat16(a * c - b * s);
        vector[rotary_pairs + i] = __float2bfloat16(b * c + a * s);
    }
    if (normalize) {
        for (int i = threadIdx.x + 2 * rotary_pairs; i < head_dim; i += blockDim.x) {
            vector[i] = __float2bfloat16(bf16_float(vector[i]) * inv * bf16_float(norm_weight[i]));
        }
    }
    __syncthreads();
    if (attention_scale != 1.0f) {
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            vector[i] = __float2bfloat16(bf16_float(vector[i]) * attention_scale);
        }
    }
    (void)section2;
}

void launch_dynamic_mrope_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim,
    const int32_t* positions, int section0, int section1, int section2,
    bool interleaved, float rope_theta, float rotary_fraction, float eps,
    bool normalize, CudaRopeScaling scaling, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    const int pairs = static_cast<int>(static_cast<float>(head_dim) * rotary_fraction) / 2;
    const float query_attention_scale = scaling.kind == 3
        ? scaling.attention_factor * scaling.attention_factor
        : 1.0f;
    dynamic_mrope_qk_norm_rope_kernel<<<q_heads, threads, 0, stream>>>(
        q, q_norm, 1, q_heads, head_dim, positions, section0, section1, section2,
        interleaved, rope_theta, pairs, eps, normalize, scaling, query_attention_scale);
    if (k) {
        dynamic_mrope_qk_norm_rope_kernel<<<kv_heads, threads, 0, stream>>>(
            k, k_norm, 1, kv_heads, head_dim, positions, section0, section1, section2,
            interleaved, rope_theta, pairs, eps, normalize, scaling, 1.0f);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
