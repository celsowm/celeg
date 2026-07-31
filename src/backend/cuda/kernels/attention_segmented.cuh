// Single-session attention split into fixed-size KV chunks: a partial kernel
// computes a per-chunk streaming softmax (max, denominator, weighted sum), then
// a reduce kernel rescales and combines the chunks.
//
// This trades one extra kernel launch and a scratch buffer for parallelism
// across the KV axis, which is what long contexts need -- a single-pass kernel
// gives only q_heads blocks of work regardless of sequence length.
//
// The decode reduce kernel is shared by the bf16 and int8 partial kernels: the
// partials it consumes are plain floats, so the KV cache precision has already
// been erased by the time it runs.

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
    CELEG_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom, partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(stream);
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
    CELEG_KERNEL_DEBUG_SYNC(stream);
    gqa_prefill_segment_reduce_kernel<<<rows * q_heads, threads, 0, stream>>>(
        out, rows, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(stream);
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
    CELEG_KERNEL_DEBUG_SYNC(stream);
    gqa_decode_segment_reduce_kernel<<<q_heads, threads, 0, stream>>>(
        out, q_heads, head_dim, chunks, partial_max, partial_denom,
        partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
