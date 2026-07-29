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

// Maximum head_dim this file's warp-only decode-attention kernels support
// (32 lanes * kMaxHeadDimPerLane each); LFM2/LFM2.5 head dims (64-128) are
// well within this.
constexpr int kMaxHeadDimPerLane = 8;

// One warp handles one (query row, query head) pair for the whole KV loop,
// with no block-wide synchronization at all: the Q.K dot product is reduced
// with warp_sum + a __shfl_sync broadcast (every lane ends up with the same
// scalar), so every lane can independently recompute the online-softmax
// running max/denominator from that broadcast value and keep its own slice
// of the V-weighted accumulator in registers. Contrast with the block-wide
// design below (attention_dot + block_sum), which pays two __syncthreads()
// per KV token; for a 512-token context that is over a thousand block-wide
// barriers per decode step, serializing every warp in the block against the
// slowest one on every single token.
__device__ __forceinline__ float warp_broadcast_sum(float partial) {
    return __shfl_sync(0xffffffffu, warp_sum(partial), 0);
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

// Warp-only online-softmax decode/prefill-fallback attention. One warp (32
// lanes, blockDim.x == 32) per (query_row, query_head): see
// warp_broadcast_sum above for why this has no __syncthreads() at all.
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
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = 0; token < seq_len; ++token) {
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

// Warp-only variant of gqa_decode_online_kernel for int8 KV cache; see
// warp_broadcast_sum above for the no-__syncthreads() reduction strategy.
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
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int token = 0; token < seq_len; ++token) {
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

// Warp-only variant: one warp (blockDim.x == 32) per (query_head, chunk).
// This is the actual hot decode-attention path once a context crosses
// attention_auto_threshold (segmented mode); see warp_broadcast_sum above.
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
        for (int d = lane; d < head_dim; d += 32) partial_accum[accum_base + d] = 0.0f;
        return;
    }

    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;

    for (int token = begin; token < end; ++token) {
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
        const float beta_value = expf(score - next_max);
        denominator = denominator * alpha + beta_value;

        const __nv_bfloat16* value = value_cache +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        int idx = 0;
        for (int d = lane; d < head_dim; d += 32, ++idx) {
            accumulator[idx] = accumulator[idx] * alpha + bf16_float(value[d]) * beta_value;
        }
        running_max = next_max;
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    int idx = 0;
    for (int d = lane; d < head_dim; d += 32, ++idx) {
        partial_accum[accum_base + d] = accumulator[idx];
    }
}

// Warp-only variant of the segmented partial kernel for int8 KV cache.
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
        for (int d = lane; d < head_dim; d += 32) partial_accum[accum_base + d] = 0.0f;
        return;
    }
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q + static_cast<size_t>(query_head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;

    for (int token = begin; token < end; ++token) {
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
        const float beta_value = expf(score - next_max);
        denominator = denominator * alpha + beta_value;

        const int8_t* value = value_cache + scale_index * head_dim;
        const float value_scale = value_scales[scale_index];
        int idx = 0;
        for (int d = lane; d < head_dim; d += 32, ++idx) {
            accumulator[idx] = accumulator[idx] * alpha +
                static_cast<float>(value[d]) * value_scale * beta_value;
        }
        running_max = next_max;
    }
    if (lane == 0) {
        partial_max[partial_index] = running_max;
        partial_denom[partial_index] = denominator;
    }
    int idx = 0;
    for (int d = lane; d < head_dim; d += 32, ++idx) {
        partial_accum[accum_base + d] = accumulator[idx];
    }
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

// Prefill counterpart of gqa_decode_segment_partial_kernel: adds a `row`
// axis (each row is causally clamped to seq_len = row + 1) so the same
// chunked online-softmax accumulation can run for every query position in
// a batched prefill instead of one query at a time.
__global__ void gqa_prefill_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, int rows, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum) {
    const int total_per_row = q_heads * chunks;
    const int flat = blockIdx.x;
    const int row = flat / total_per_row;
    const int rem = flat % total_per_row;
    const int query_head = rem / chunks;
    const int chunk = rem % chunks;
    if (row >= rows || query_head >= q_heads) return;

    const int seq_len = row + 1;
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

__global__ void gqa_prefill_segment_reduce_kernel(
    __nv_bfloat16* out, int rows, int q_heads, int head_dim, int chunks,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    const int lane = threadIdx.x;
    if (row >= rows) return;
    const size_t base = (static_cast<size_t>(row) * q_heads + query_head) * chunks;
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
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
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
    gqa_decode_online_kernel<<<q_heads, 32, 0, stream>>>(
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
    gqa_decode_online_kernel<<<q_heads, 32, 0, stream>>>(
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
    gqa_decode_segment_partial_kernel<<<q_heads * chunks, 32, 0, stream>>>(
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
    gqa_decode_online_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, out, rows, 0, nullptr, 2,
        q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_segmented(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    gqa_prefill_segment_partial_kernel<<<rows * q_heads * chunks, threads, 0, stream>>>(
        q, key_cache, value_cache, rows, q_heads, kv_heads, head_dim,
        chunk_tokens, chunks, partial_max, partial_denom, partial_accum);
    LFM_KERNEL_DEBUG_SYNC(stream);
    gqa_prefill_segment_reduce_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
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
    gqa_decode_online_int8_kernel<<<q_heads, 32, 0, stream>>>(
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
    gqa_decode_online_int8_kernel<<<q_heads, 32, 0, stream>>>(
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
    gqa_decode_segment_partial_int8_kernel<<<q_heads * chunks, 32, 0, stream>>>(
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
    gqa_decode_online_int8_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, rows,
        0, nullptr, 2, q_heads, kv_heads, head_dim);
    LFM_KERNEL_DEBUG_SYNC(stream);
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

// ---------------------------------------------------------------------------
// Batched-GEMM causal prefill attention.
//
// The per-(row,head) kernels above compute one scalar dot product per KV
// token per thread-block, with no tensor cores - fine for decode (one query
// against a long KV history) but the wrong shape for prefill, where every
// row attends to a *different* prefix of the *same* rows. That is exactly
// the shape cuBLAS batched GEMM wants: for each head, QK^T and softmax(.)V
// are dense matrix multiplies. This path computes raw scores with one
// strided-batched GEMM per KV-head group (broadcasting the shared K/V
// across the q_heads/kv_heads queries in that GQA group via strideB/strideA
// = 0), a causal softmax kernel, then a second strided-batched GEMM for the
// value contraction. Two cuBLAS calls per KV head instead of one scalar
// dot product per KV token per row.
// ---------------------------------------------------------------------------

// One block per (head, row). Reduces over the causal prefix [0, row] twice
// (max, then sum of exp) using the existing block_max/block_sum reductions,
// then writes a full-width BF16 probability row (zero past `row`) so the
// following dense PV GEMM naturally ignores masked positions.
__global__ void causal_softmax_kernel(const float* __restrict__ scores,
                                      __nv_bfloat16* __restrict__ probs,
                                      int rows, int q_heads) {
    const int flat = blockIdx.x;
    const int head = flat / rows;
    const int row = flat % rows;
    if (head >= q_heads) return;
    const size_t base = (static_cast<size_t>(head) * rows + row) * static_cast<size_t>(rows);
    const int valid = row + 1;
    const int lane = threadIdx.x;
    const int threads = blockDim.x;

    __shared__ float warp_scratch[32];
    __shared__ float block_value;

    float local_max = -FLT_MAX;
    for (int c = lane; c < valid; c += threads) {
        local_max = fmaxf(local_max, scores[base + c]);
    }
    const float row_max = block_max(local_max, warp_scratch, &block_value);

    float local_sum = 0.0f;
    for (int c = lane; c < valid; c += threads) {
        local_sum += expf(scores[base + c] - row_max);
    }
    const float row_sum = block_sum(local_sum, warp_scratch, &block_value);
    const float inv_sum = 1.0f / row_sum;

    for (int c = lane; c < rows; c += threads) {
        const float p = c < valid ? expf(scores[base + c] - row_max) * inv_sum : 0.0f;
        probs[base + c] = __float2bfloat16(p);
    }
}

void launch_causal_softmax(const float* scores, __nv_bfloat16* probs,
                           int rows, int q_heads, cudaStream_t stream) {
    const int threads = rows < 256 ? ((rows + 31) / 32) * 32 : 256;
    causal_softmax_kernel<<<rows * q_heads, max(threads, 32), 0, stream>>>(
        scores, probs, rows, q_heads);
    LFM_KERNEL_CHECK();
}

void launch_gqa_prefill_gemm(
    cublasHandle_t cublas, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* out, float* scores_scratch,
    __nv_bfloat16* probs_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int q_width, int kv_width, int out_width,
    cudaStream_t stream) {
    const int group = q_heads / kv_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float zero = 0.0f;
    const long long rows_sq = static_cast<long long>(rows) * rows;

    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* k_base = k + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* q_base = q +
            static_cast<size_t>(kv_head) * group * head_dim;
        float* scores_base = scores_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        // scores[rows,rows] (row-major) = Q[rows,head_dim] @ K^T[head_dim,rows],
        // scaled by 1/sqrt(head_dim). Same col-major transpose recipe as
        // GemmDispatcher::linear_cublas, batched over the `group` queries
        // that share this KV head (K is broadcast: strideA = 0).
        LFM_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, rows, head_dim,
            &scale,
            k_base, CUDA_R_16BF, kv_width, 0,
            q_base, CUDA_R_16BF, q_width, head_dim,
            &zero,
            scores_base, CUDA_R_32F, rows, rows_sq,
            group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    launch_causal_softmax(scores_scratch, probs_scratch, rows, q_heads, stream);

    const float one = 1.0f;
    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* v_base = v + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* probs_base = probs_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        __nv_bfloat16* out_base = out +
            static_cast<size_t>(kv_head) * group * head_dim;
        // out[rows,head_dim] (row-major) = P[rows,rows] @ V[rows,head_dim].
        // Row-major-via-col-major no-transpose recipe, batched over the
        // group's queries (V is broadcast: strideA = 0 in this call's A
        // role, which is our math "B"/V).
        LFM_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, rows, rows,
            &one,
            v_base, CUDA_R_16BF, kv_width, 0,
            probs_base, CUDA_R_16BF, rows, rows_sq,
            &zero,
            out_base, CUDA_R_16BF, out_width, head_dim,
            group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}
