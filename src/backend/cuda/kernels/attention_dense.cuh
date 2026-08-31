
__global__ void gqa_decode_strict_kernel(const __nv_bfloat16* q,
                                         const __nv_bfloat16* key_cache,
                                         const __nv_bfloat16* value_cache,
                                         __nv_bfloat16* out,
                                         int rows,
                                         int seq_len_value,
                                         const int32_t* position_pointer,
                                         int mode,
                                         int q_heads,
                                         int kv_heads,
                                         int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const bool prefill = mode >= 2;
    const int query_row = prefill ? block / q_heads : 0;
    const int query_head = prefill ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 1 ? *position_pointer + 1 :
        (mode == 2 ? query_row + 1 :
         (mode == 4 && query_row >= seq_len_value ? query_row + 1 : seq_len_value));
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }

    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator += probability * bf16_float(value[lane]);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_kernel(const __nv_bfloat16* q,
                                         const __nv_bfloat16* key_cache,
                                         const __nv_bfloat16* value_cache,
                                         __nv_bfloat16* out,
                                         int rows,
                                         int seq_len_value,
                                         const int32_t* position_pointer,
                                         int mode,
                                         int q_heads,
                                         int kv_heads,
                                         int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;

    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            partial += bf16_float(query[d]) * bf16_float(key[d]);
        }
        const float dot = warp_broadcast_sum(partial);
        const float score = dot * scale;
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;

        const __nv_bfloat16* value = value_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        int idx = 0;
        for (int d = lane; d < head_dim; d += 32, ++idx) {
            accumulator[idx] = accumulator[idx] * alpha + bf16_float(value[d]) * beta;
        }
        running_max = next_max;
    }

    __nv_bfloat16* out_row = out +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    int idx = 0;
    for (int d = lane; d < head_dim; d += 32, ++idx) {
        out_row[d] = __float2bfloat16(accumulator[idx] / denominator);
    }
}

__global__ void gqa_decode_strict_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int seq_len_value, const int32_t* position_pointer, int mode,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const bool prefill = mode >= 2;
    const int query_row = prefill ? block / q_heads : 0;
    const int query_head = prefill ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 1 ? *position_pointer + 1 :
        (mode == 2 ? query_row + 1 :
         (mode == 4 && query_row >= seq_len_value ? query_row + 1 : seq_len_value));
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }
    float accumulator = 0.0f;
    for (int token = first_token; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator += probability *
                (static_cast<float>(value[lane]) * value_scales[scale_index]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int seq_len_value, const int32_t* position_pointer, int mode,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = first_token; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float key_scale = key_scales[scale_index];
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            partial += bf16_float(query[d]) * (static_cast<float>(key[d]) * key_scale);
        }
        const float dot = warp_broadcast_sum(partial);
        const float score = dot * scale;
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;

        const int8_t* value = value_cache + scale_index * head_dim;
        const float value_scale = value_scales[scale_index];
        int idx = 0;
        for (int d = lane; d < head_dim; d += 32, ++idx) {
            accumulator[idx] = accumulator[idx] * alpha +
                static_cast<float>(value[d]) * value_scale * beta;
        }
        running_max = next_max;
    }

    __nv_bfloat16* out_row = out +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
    int idx = 0;
    for (int d = lane; d < head_dim; d += 32, ++idx) {
        out_row[d] = __float2bfloat16(accumulator[idx] / denominator);
    }
}

void launch_gqa_decode_strict(const GqaContiguousArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_decode_strict_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, 1,
        args.extent.seq_len, nullptr, 0,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_strict_device(const GqaContiguousArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_decode_strict_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, 1, 0,
        args.extent.position, 1,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_online(const GqaContiguousArgs& args) {
    gqa_decode_online_kernel<<<args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, 1,
        args.extent.seq_len, nullptr, 0,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_online_device(const GqaContiguousArgs& args) {
    gqa_decode_online_kernel<<<args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, 1, 0,
        args.extent.position, 1,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_strict(const GqaContiguousArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const int rows = args.extent.rows;
    const int mode = args.extent.seq_len <= 0 ? 2 :
        (args.extent.seq_len < rows ? 4 : 3);
    gqa_decode_strict_kernel<<<rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, rows,
        args.extent.seq_len, nullptr, mode,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_online(const GqaContiguousArgs& args) {
    const int rows = args.extent.rows;
    gqa_decode_online_kernel<<<rows * args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, rows, 0, nullptr, 2,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_strict_int8(const GqaContiguousInt8Args& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_decode_strict_int8_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, 1, args.extent.seq_len, nullptr, 0,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_online_int8(const GqaContiguousInt8Args& args) {
    gqa_decode_online_int8_kernel<<<args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, 1, args.extent.seq_len, nullptr, 0,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_strict_int8_device(const GqaContiguousInt8Args& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    gqa_decode_strict_int8_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, 1, 0, args.extent.position, 1,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_online_int8_device(const GqaContiguousInt8Args& args) {
    gqa_decode_online_int8_kernel<<<args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, 1, 0, args.extent.position, 1,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_strict_int8(const GqaContiguousInt8Args& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const int rows = args.extent.rows;
    const int mode = args.extent.seq_len <= 0 ? 2 :
        (args.extent.seq_len < rows ? 4 : 3);
    gqa_decode_strict_int8_kernel<<<rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, rows, args.extent.seq_len, nullptr, mode,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_online_int8(const GqaContiguousInt8Args& args) {
    const int rows = args.extent.rows;
    gqa_decode_online_int8_kernel<<<rows * args.geometry.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, rows, 0, nullptr, 2,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        args.geometry.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
