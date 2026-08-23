
// The decode chunk count is fixed at plan time from the context *capacity*
// (execution_plan.cpp), because a CUDA-graph-captured decode step needs a
// static grid. Only the chunks covering [first_token, seq_len) hold keys, so
// the rest early-out without touching the partial buffers at all, and the
// reduce kernel below derives the same live range from the device-side
// position rather than reading every chunk back.
__device__ inline void decode_live_chunk_range(int seq_len, int chunk_tokens,
                                               int chunks, int sliding_window,
                                               int* first, int* last) {
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    *first = min(first_token / chunk_tokens, chunks);
    *last = min((seq_len + chunk_tokens - 1) / chunk_tokens, chunks);
}

__global__ void gqa_decode_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int begin = max(chunk * chunk_tokens, first_token);
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    // No zero-fill: the reduce kernel skips this chunk entirely.
    if (begin >= end) return;

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

__global__ void gqa_decode_segment_partial_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / chunks;
    const int chunk = flat % chunks;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int begin = max(chunk * chunk_tokens, first_token);
    const int end = min(begin + chunk_tokens, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * chunks + chunk;
    const size_t accum_base = partial_index * head_dim;
    // No zero-fill: the reduce kernel skips this chunk entirely.
    if (begin >= end) return;
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
    __nv_bfloat16* out, const int32_t* position, int q_heads, int head_dim,
    int chunk_tokens, int chunks, int sliding_window,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_head = blockIdx.x;
    const int lane = threadIdx.x;
    if (query_head >= q_heads) return;
    int first_chunk = 0;
    int last_chunk = chunks;
    decode_live_chunk_range(*position + 1, chunk_tokens, chunks, sliding_window,
                            &first_chunk, &last_chunk);
    const size_t base = static_cast<size_t>(query_head) * chunks;
    float global_max = -FLT_MAX;
    for (int chunk = first_chunk; chunk < last_chunk; ++chunk) {
        global_max = fmaxf(global_max, partial_max[base + chunk]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int chunk = first_chunk; chunk < last_chunk; ++chunk) {
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

__global__ void gqa_prefill_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, int rows, int q_heads, int kv_heads,
    int head_dim, int chunk_tokens, int chunks, int sliding_window,
    float* partial_max,
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

void launch_gqa_decode_segmented_device(const GqaSegmentedArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const AttentionSegmentation& seg = args.segmentation;
    gqa_decode_segment_partial_kernel<<<args.geometry.q_heads * seg.chunks, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        seg.chunk_tokens, seg.chunks, args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
    gqa_decode_segment_reduce_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.out, args.extent.position, args.geometry.q_heads,
        args.geometry.head_dim, seg.chunk_tokens, seg.chunks,
        args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_segmented(const GqaSegmentedArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const AttentionSegmentation& seg = args.segmentation;
    const int rows = args.extent.rows;
    gqa_prefill_segment_partial_kernel<<<rows * args.geometry.q_heads * seg.chunks, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, rows,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        seg.chunk_tokens, seg.chunks, args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
    gqa_prefill_segment_reduce_kernel<<<rows * args.geometry.q_heads, threads, 0, args.stream>>>(
        args.out, rows, args.geometry.q_heads, args.geometry.head_dim,
        seg.chunks, seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_segmented_int8_device(const GqaSegmentedInt8Args& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const AttentionSegmentation& seg = args.segmentation;
    gqa_decode_segment_partial_int8_kernel<<<args.geometry.q_heads * seg.chunks, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        seg.chunk_tokens, seg.chunks, args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
    gqa_decode_segment_reduce_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.out, args.extent.position, args.geometry.q_heads,
        args.geometry.head_dim, seg.chunk_tokens, seg.chunks,
        args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
