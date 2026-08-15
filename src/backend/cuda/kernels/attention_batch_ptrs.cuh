
__global__ void gqa_decode_online_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const __nv_bfloat16* row_keys = key_cache[row];
    const __nv_bfloat16* row_values = value_cache[row];
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = row_values +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha + bf16_float(value[lane]) * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_strict_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const __nv_bfloat16* row_keys = key_cache[row];
    const __nv_bfloat16* row_values = value_cache[row];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const __nv_bfloat16* key = row_keys +
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
        const __nv_bfloat16* key = row_keys +
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
        const __nv_bfloat16* key = row_keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = row_values +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator += probability * bf16_float(value[lane]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

__global__ void gqa_decode_online_int8_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const int8_t* row_keys = key_cache[row];
    const int8_t* row_values = value_cache[row];
    const float* row_key_scales = key_scales[row];
    const float* row_value_scales = value_scales[row];
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = first_token; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = row_values + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * row_value_scales[scale_index] * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_strict_int8_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const int8_t* row_keys = key_cache[row];
    const int8_t* row_values = value_cache[row];
    const float* row_key_scales = key_scales[row];
    const float* row_value_scales = value_scales[row];
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
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
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
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
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
        const int8_t* key = row_keys + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, row_key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = row_values + scale_index * head_dim;
            accumulator += probability *
                (static_cast<float>(value[lane]) * row_value_scales[scale_index]);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

void launch_gqa_decode_batch_ptrs(const GqaBatchPtrArgs& args) {
    const GqaGeometry& g = args.geometry;
    const int threads = attention_threads(g.head_dim);
    if (args.fast) {
        gqa_decode_online_batch_ptrs_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.out, args.positions,
            args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_strict_batch_ptrs_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.out, args.positions,
            args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_int8_batch_ptrs(const GqaBatchPtrInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const int threads = attention_threads(g.head_dim);
    if (args.fast) {
        gqa_decode_online_int8_batch_ptrs_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
            args.kv.value_scales, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_strict_int8_batch_ptrs_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
            args.kv.value_scales, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
