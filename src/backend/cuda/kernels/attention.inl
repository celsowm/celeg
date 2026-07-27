__global__ void store_kv_kernel(const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                __nv_bfloat16* key_cache,
                                __nv_bfloat16* value_cache,
                                int rows,
                                int width,
                                int position_value,
                                const int32_t* position_pointer,
                                int mode) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    key_cache[static_cast<size_t>(target_row) * width + column] = k[index];
    value_cache[static_cast<size_t>(target_row) * width + column] = v[index];
}

__global__ void store_kv_int8_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_cache, int8_t* value_cache,
    float* key_scales, float* value_scales, int rows, int kv_heads,
    int head_dim, int position_value, const int32_t* position_pointer,
    int mode) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows || head >= kv_heads) return;
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t cache_base =
        (static_cast<size_t>(target_row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index =
            static_cast<size_t>(target_row) * kv_heads + head;
        key_scales[scale_index] = key_scale;
        value_scales[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

__device__ float attention_dot_int8(const __nv_bfloat16* query,
                                    const int8_t* key, float key_scale,
                                    int head_dim, float* warp_sums,
                                    float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) *
            (static_cast<float>(key[d]) * key_scale);
    }
    return block_sum(partial, warp_sums, total);
}

__device__ float attention_dot(const __nv_bfloat16* query,
                               const __nv_bfloat16* key,
                               int head_dim,
                               float* warp_sums,
                               float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) * bf16_float(key[d]);
    }
    return block_sum(partial, warp_sums, total);
}

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
                                         int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
                                         int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;

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

    for (int token = 0; token < seq_len; ++token) {
        const __nv_bfloat16* key = key_cache +
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
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha + bf16_float(value[lane]) * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }

    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}


__global__ void gqa_decode_strict_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int seq_len_value, const int32_t* position_pointer, int mode,
    int q_heads, int kv_heads, int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    int q_heads, int kv_heads, int head_dim) {
    const int block = blockIdx.x;
    const int query_row = mode == 2 ? block / q_heads : 0;
    const int query_head = mode == 2 ? block % q_heads : block;
    if (query_row >= rows || query_head >= q_heads) return;
    const int seq_len = mode == 2 ? query_row + 1 :
        (mode == 1 ? *position_pointer + 1 : seq_len_value);
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(query_row) * q_heads + query_head) * head_dim;
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
    for (int token = 0; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
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
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * value_scales[scale_index] * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(query_row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void gqa_decode_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }

    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;

    for (int token = begin; token < end; ++token) {
        const __nv_bfloat16* key = key_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        const float dot = attention_dot(query, key, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const __nv_bfloat16* value = value_cache +
                (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            accumulator = accumulator * alpha +
                bf16_float(value[lane]) * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_partial_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = key_cache + scale_index * head_dim;
        const float dot = attention_dot_int8(query, key, key_scales[scale_index],
                                             head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const int8_t* value = value_cache + scale_index * head_dim;
            accumulator = accumulator * alpha +
                static_cast<float>(value[lane]) * value_scales[scale_index] * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_reduce_kernel(
    __nv_bfloat16* out, int q_heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_head = blockIdx.x;
    const int lane = threadIdx.x;
    if (query_head >= q_heads) return;
    const size_t base = static_cast<size_t>(query_head) * chunks;
    float global_max = -FLT_MAX;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float local_denom = partial_denom[base + chunk];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + chunk] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            const size_t accum_index =
                (base + chunk) * static_cast<size_t>(head_dim) + lane;
            accumulator += partial_accum[accum_index] * factor;
        }
    }
    if (lane < head_dim) {
        out[static_cast<size_t>(query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

__global__ void store_kv_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width) {
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const size_t target = static_cast<size_t>(positions[row]) * kv_width + column;
    key_cache[row][target] = k[index];
    value_cache[row][target] = v[index];
}

__global__ void store_kv_int8_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t scale_index =
        static_cast<size_t>(positions[row]) * kv_heads + head;
    const size_t cache_base = scale_index * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        key_scales[row][scale_index] = key_scale;
        value_scales[row][scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

__global__ void gqa_decode_online_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
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
    for (int token = 0; token < seq_len; ++token) {
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
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
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
    for (int token = 0; token < seq_len; ++token) {
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
    int head_dim) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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
    for (int token = 0; token < seq_len; ++token) {
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

void launch_store_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                     __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
                     int position, int kv_width, cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, position, nullptr, 0);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_device(const __nv_bfloat16* k, const __nv_bfloat16* v,
                            __nv_bfloat16* key_cache,
                            __nv_bfloat16* value_cache,
                            const int32_t* position, int kv_width,
                            cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, 0, position, 1);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_prefill(const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* key_cache,
                             __nv_bfloat16* value_cache,
                             int rows, int kv_width, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, rows, kv_width, 0, nullptr, 2);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int8_t* key_cache, int8_t* value_cache,
                          float* key_scales, float* value_scales,
                          int position, int kv_heads, int head_dim,
                          cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, position, nullptr, 0);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_device(const __nv_bfloat16* k,
                                 const __nv_bfloat16* v,
                                 int8_t* key_cache, int8_t* value_cache,
                                 float* key_scales, float* value_scales,
                                 const int32_t* position, int kv_heads,
                                 int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, 0, position, 1);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_prefill(const __nv_bfloat16* k,
                                  const __nv_bfloat16* v,
                                  int8_t* key_cache, int8_t* value_cache,
                                  float* key_scales, float* value_scales,
                                  int rows, int kv_heads, int head_dim,
                                  cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, rows,
        kv_heads, head_dim, 0, nullptr, 2);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, seq_len, nullptr, 0,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, 0, position, 1,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, seq_len, nullptr, 0,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, 1, 0, position, 1,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_segmented_device(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_segment_partial_kernel<<<q_heads * chunks, threads, 0, stream>>>(
        q, key_cache, value_cache, position, q_heads, kv_heads, head_dim,
        chunk_tokens, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_strict(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, rows, 0, nullptr, 2,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_online(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, out, rows, 0, nullptr, 2,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        seq_len, nullptr, 0, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        seq_len, nullptr, 0, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_strict_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        0, position, 1, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_online_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, 1,
        0, position, 1, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_segmented_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_segment_partial_int8_kernel<<<q_heads * chunks, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, position,
        q_heads, kv_heads, head_dim, chunk_tokens, chunks, partial_max,
        partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_strict_int8_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, rows,
        0, nullptr, 2, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_online_int8_kernel<<<rows * q_heads, threads, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, rows,
        0, nullptr, 2, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

__device__ __forceinline__ size_t paged_vector_offset(
    uint32_t page, int attention_slot, int in_page, int head, int dim,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    return (((((static_cast<size_t>(page) * attention_layers + attention_slot) *
               page_tokens + in_page) * kv_heads + head) * head_dim) + dim);
}

__device__ __forceinline__ size_t paged_scale_offset(
    uint32_t page, int attention_slot, int in_page, int head,
    int page_tokens, int attention_layers, int kv_heads) {
    return ((((static_cast<size_t>(page) * attention_layers + attention_slot) *
              page_tokens + in_page) * kv_heads) + head);
}

__global__ void store_kv_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    const size_t kv_width = static_cast<size_t>(kv_heads) * head_dim;
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const int head = column / head_dim;
    const int dim = column % head_dim;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t target = paged_vector_offset(page, attention_slot, in_page,
                                               head, dim, page_tokens,
                                               attention_layers, kv_heads,
                                               head_dim);
    key_pool[target] = k[index];
    value_pool[target] = v[index];
}

__global__ void store_kv_int8_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index = paged_scale_offset(
            page, attention_slot, in_page, head, page_tokens,
            attention_layers, kv_heads);
        key_scale_pool[scale_index] = key_scale;
        value_scale_pool[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const size_t target = paged_vector_offset(
            page, attention_slot, in_page, head, d, page_tokens,
            attention_layers, kv_heads, head_dim);
        key_pool[target] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_pool[target] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

template <bool Strict>
__global__ void gqa_decode_paged_batch_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if constexpr (Strict) {
        if (lane == 0) maximum = -FLT_MAX;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                maximum = fmaxf(maximum, score);
            }
            __syncthreads();
        }
        if (lane == 0) denominator = 0.0f;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                denominator += expf(score - maximum);
            }
            __syncthreads();
        }
        float accumulator = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
                probability = rounded_bf16_float(expf(score - maximum) / denominator);
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator += probability * bf16_float(value_pool[offset]);
            }
            __syncthreads();
        }
        if (lane < head_dim) {
            out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
                __float2bfloat16(accumulator);
        }
    } else {
        float running_max = -FLT_MAX;
        float denom = 0.0f;
        float accumulator = 0.0f;
        __shared__ float alpha;
        __shared__ float beta;
        __shared__ float next_max;
        __shared__ float next_denom;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    attention_layers, kv_heads, head_dim);
                local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
            }
            const float dot = block_sum(local, warp_sums, &dot_total);
            if (lane == 0) {
                const float score = dot * scale;
                next_max = fmaxf(running_max, score);
                alpha = expf(running_max - next_max);
                beta = expf(score - next_max);
                next_denom = denom * alpha + beta;
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator = accumulator * alpha + bf16_float(value_pool[offset]) * beta;
            }
            denom = next_denom;
            running_max = next_max;
            __syncthreads();
        }
        if (lane < head_dim) {
            out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
                __float2bfloat16(accumulator / denom);
        }
    }
}


__device__ float paged_int8_attention_dot(
    const __nv_bfloat16* query,
    const int8_t* key_pool,
    const float* key_scale_pool,
    uint32_t page, int attention_slot, int in_page, int kv_head,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    float* warp_sums, float* dot_total) {
    const int lane = threadIdx.x;
    const size_t scale_offset = paged_scale_offset(
        page, attention_slot, in_page, kv_head, page_tokens,
        attention_layers, kv_heads);
    const float key_scale = key_scale_pool[scale_offset];
    float local = 0.0f;
    for (int d = lane; d < head_dim; d += blockDim.x) {
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, d, page_tokens,
            attention_layers, kv_heads, head_dim);
        local += bf16_float(query[d]) * static_cast<float>(key_pool[offset]) * key_scale;
    }
    return block_sum(local, warp_sums, dot_total);
}

template <bool Strict>
__global__ void gqa_decode_int8_paged_batch_kernel(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if constexpr (Strict) {
        if (lane == 0) maximum = -FLT_MAX;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) maximum = fmaxf(maximum,
                rounded_bf16_float(rounded_bf16_float(dot) * scale));
            __syncthreads();
        }
        if (lane == 0) denominator = 0.0f;
        __syncthreads();
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) denominator += expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum);
            __syncthreads();
        }
        float accumulator = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) probability = rounded_bf16_float(expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum) / denominator);
            __syncthreads();
            if (lane < head_dim) {
                const size_t scale_offset = paged_scale_offset(
                    page, attention_slot, ip, kv_head, page_tokens,
                    attention_layers, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator += probability * static_cast<float>(value_pool[offset]) *
                               value_scale_pool[scale_offset];
            }
            __syncthreads();
        }
        if (lane < head_dim) out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    } else {
        float running_max = -FLT_MAX, denom = 0.0f, accumulator = 0.0f;
        __shared__ float alpha, beta, next_max, next_denom;
        for (int token = 0; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, attention_layers, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) {
                const float score = dot * scale;
                next_max = fmaxf(running_max, score);
                alpha = expf(running_max - next_max);
                beta = expf(score - next_max);
                next_denom = denom * alpha + beta;
            }
            __syncthreads();
            if (lane < head_dim) {
                const size_t scale_offset = paged_scale_offset(
                    page, attention_slot, ip, kv_head, page_tokens,
                    attention_layers, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    attention_layers, kv_heads, head_dim);
                accumulator = accumulator * alpha + static_cast<float>(value_pool[offset]) *
                              value_scale_pool[scale_offset] * beta;
            }
            denom = next_denom;
            running_max = next_max;
            __syncthreads();
        }
        if (lane < head_dim) out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denom);
    }
}


__global__ void gqa_decode_paged_segment_partial_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const int lp = token / page_tokens;
        const int ip = token % page_tokens;
        const uint32_t page = page_tables[
            static_cast<size_t>(row) * page_table_stride + lp];
        float local = 0.0f;
        for (int d = lane; d < head_dim; d += blockDim.x) {
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, d, page_tokens,
                attention_layers, kv_heads, head_dim);
            local += bf16_float(query[d]) * bf16_float(key_pool[offset]);
        }
        const float dot = block_sum(local, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, lane, page_tokens,
                attention_layers, kv_heads, head_dim);
            accumulator = accumulator * alpha +
                bf16_float(value_pool[offset]) * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_int8_paged_segment_partial_kernel(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int begin = chunk * chunk_tokens;
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    if (begin >= end) {
        if (lane == 0) {
            partial_max[partial_index] = -FLT_MAX;
            partial_denom[partial_index] = 0.0f;
        }
        if (lane < head_dim) partial_accum[accum_base + lane] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta_value;
    __shared__ float next_max;
    __shared__ float shared_denom;
    for (int token = begin; token < end; ++token) {
        const int lp = token / page_tokens;
        const int ip = token % page_tokens;
        const uint32_t page = page_tables[
            static_cast<size_t>(row) * page_table_stride + lp];
        const float dot = paged_int8_attention_dot(
            query, key_pool, key_scale_pool, page, attention_slot, ip,
            kv_head, page_tokens, attention_layers, kv_heads, head_dim,
            warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta_value = expf(score - next_max);
            shared_denom = denominator * alpha + beta_value;
        }
        __syncthreads();
        if (lane < head_dim) {
            const size_t scale_index = paged_scale_offset(
                page, attention_slot, ip, kv_head, page_tokens,
                attention_layers, kv_heads);
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, lane, page_tokens,
                attention_layers, kv_heads, head_dim);
            accumulator = accumulator * alpha +
                static_cast<float>(value_pool[offset]) *
                value_scale_pool[scale_index] * beta_value;
        }
        running_max = next_max;
        denominator = shared_denom;
        __syncthreads();
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    if (lane < head_dim) partial_accum[accum_base + lane] = accumulator;
}

__global__ void gqa_decode_segment_reduce_batch_kernel(
    __nv_bfloat16* out, int rows, int q_heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_index = blockIdx.x;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    const size_t base =
        (static_cast<size_t>(row) * q_heads + query_head) * chunks;
    float global_max = -FLT_MAX;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float local_denom = partial_denom[base + chunk];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + chunk] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            accumulator += partial_accum[
                (base + chunk) * static_cast<size_t>(head_dim) + lane] * factor;
        }
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

void launch_store_kv_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_batch_ptrs_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, positions, rows, kv_width);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_batch_ptrs_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales,
        positions, rows, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_batch_ptrs(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        gqa_decode_online_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, out, positions,
            rows, q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_strict_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, out, positions,
            rows, q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_int8_batch_ptrs(
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
    int head_dim,
    bool fast,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    if (fast) {
        gqa_decode_online_int8_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, key_scales, value_scales,
            out, positions, rows, q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_strict_int8_batch_ptrs_kernel<<<rows * q_heads, threads, 0, stream>>>(
            q, key_cache, value_cache, key_scales, value_scales,
            out, positions, rows, q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * kv_heads * head_dim;
    store_kv_paged_batch_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        k, v, key_pool, value_pool, page_tables, page_table_stride,
        positions, rows, attention_slot, page_tokens, attention_layers,
        kv_heads, head_dim);
    LFM_KERNEL_CHECK();
}

void launch_store_kv_int8_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream) {
    store_kv_int8_paged_batch_kernel<<<rows * kv_heads, 64, 0, stream>>>(
        k, v, key_pool, value_pool, key_scale_pool, value_scale_pool,
        page_tables, page_table_stride, positions, rows, attention_slot,
        page_tokens, attention_layers, kv_heads, head_dim);
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_paged_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream) {
    if (fast) {
        gqa_decode_paged_batch_kernel<false><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, page_tables, page_table_stride, out,
            positions, rows, attention_slot, page_tokens, attention_layers,
            q_heads, kv_heads, head_dim);
    } else {
        gqa_decode_paged_batch_kernel<true><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, page_tables, page_table_stride, out,
            positions, rows, attention_slot, page_tokens, attention_layers,
            q_heads, kv_heads, head_dim);
    }
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream) {
    if (fast) {
        gqa_decode_int8_paged_batch_kernel<false><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, key_scale_pool, value_scale_pool,
            page_tables, page_table_stride, out, positions, rows,
            attention_slot, page_tokens, attention_layers, q_heads,
            kv_heads, head_dim);
    } else {
        gqa_decode_int8_paged_batch_kernel<true><<<rows * q_heads, 64, 0, stream>>>(
            q, key_pool, value_pool, key_scale_pool, value_scale_pool,
            page_tables, page_table_stride, out, positions, rows,
            attention_slot, page_tokens, attention_layers, q_heads,
            kv_heads, head_dim);
    }
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_paged_segmented_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_paged_segment_partial_kernel<<<rows * q_heads * chunks, threads, 0, stream>>>(
        q, key_pool, value_pool, page_tables, page_table_stride, positions,
        rows, attention_slot, page_tokens, attention_layers, q_heads,
        kv_heads, head_dim, chunk_tokens, chunks, partial_max,
        partial_denom, partial_accum);
    LFM_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_segmented_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_decode_int8_paged_segment_partial_kernel<<<rows * q_heads * chunks, threads, 0, stream>>>(
        q, key_pool, value_pool, key_scale_pool, value_scale_pool,
        page_tables, page_table_stride, positions, rows, attention_slot,
        page_tokens, attention_layers, q_heads, kv_heads, head_dim,
        chunk_tokens, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    LFM_KERNEL_CHECK();
}

