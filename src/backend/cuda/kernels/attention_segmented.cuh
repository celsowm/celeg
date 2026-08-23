
// Decode attention divides the live KV range [first_token, seq_len) into equal
// segments. The grid width and the partial buffers are sized once from the
// context capacity (decode_attention_segments), but how many of those segments
// are actually used is decided here, per step, from the device-side position --
// so the work tracks the sequence that exists rather than the one that was
// allocated for, and the grid stays static for CUDA graph capture. Both the
// partial and the reduce kernel call this so they agree on which segments were
// written.
struct DecodeSegmentPlan {
    int first_token;
    int active;         // segments actually used this step
    int tokens_each;
};

__device__ inline DecodeSegmentPlan decode_segment_plan(int seq_len, int segments,
                                                        int min_segments,
                                                        int sliding_window) {
    DecodeSegmentPlan plan;
    plan.first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int live = max(1, seq_len - plan.first_token);
    // Splitting further than kDecodeTokensPerSegment shortens the partial
    // kernel's token loop but lengthens the reduce kernel's loop over
    // segments, and past this point the reduce loses; below min_segments there
    // are not enough blocks to fill the device, so a short sequence is spread
    // as widely as it has tokens.
    const int wanted = (live + kDecodeTokensPerSegment - 1) / kDecodeTokensPerSegment;
    plan.active = min(segments, min(live, max(min_segments, wanted)));
    plan.tokens_each = (live + plan.active - 1) / plan.active;
    return plan;
}

__global__ void gqa_decode_segment_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, const int32_t* position,
    int q_heads, int kv_heads, int head_dim, int segments, int min_segments,
    int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / segments;
    const int segment = flat % segments;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const DecodeSegmentPlan plan =
        decode_segment_plan(seq_len, segments, min_segments, sliding_window);
    if (segment >= plan.active) return;
    const int begin = plan.first_token + segment * plan.tokens_each;
    const int end = min(begin + plan.tokens_each, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * segments + segment;
    const size_t accum_base = partial_index * head_dim;
    // No zero-fill: the reduce kernel derives the same range and skips this one.
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
    int q_heads, int kv_heads, int head_dim, int segments, int min_segments,
    int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int query_head = flat / segments;
    const int segment = flat % segments;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const DecodeSegmentPlan plan =
        decode_segment_plan(seq_len, segments, min_segments, sliding_window);
    if (segment >= plan.active) return;
    const int begin = plan.first_token + segment * plan.tokens_each;
    const int end = min(begin + plan.tokens_each, seq_len);
    const int lane = threadIdx.x;
    const size_t partial_index = static_cast<size_t>(query_head) * segments + segment;
    const size_t accum_base = partial_index * head_dim;
    // No zero-fill: the reduce kernel derives the same range and skips this one.
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
    int segments, int min_segments, int sliding_window,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    const int query_head = blockIdx.x;
    const int lane = threadIdx.x;
    if (query_head >= q_heads) return;
    const int seq_len = *position + 1;
    const DecodeSegmentPlan plan =
        decode_segment_plan(seq_len, segments, min_segments, sliding_window);
    // tokens_each is a ceiling, so the tail segments of `active` can be empty;
    // they are exactly the ones the partial kernel skipped, and since `begin`
    // is monotonic the first empty one ends the range.
    int written = plan.active;
    for (int segment = 0; segment < plan.active; ++segment) {
        if (plan.first_token + segment * plan.tokens_each >= seq_len) {
            written = segment;
            break;
        }
    }
    const size_t base = static_cast<size_t>(query_head) * segments;
    float global_max = -FLT_MAX;
    for (int segment = 0; segment < written; ++segment) {
        global_max = fmaxf(global_max, partial_max[base + segment]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int segment = 0; segment < written; ++segment) {
        const float local_denom = partial_denom[base + segment];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + segment] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            const size_t accum_index =
                (base + segment) * static_cast<size_t>(head_dim) + lane;
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

namespace {
int decode_attention_blocks_target() {
    static const int value = [] {
        int device = 0;
        int multiprocessors = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&multiprocessors,
                                   cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess) {
            multiprocessors = 1;
        }
        // Each (head, segment) is one warp-sized block, and the token loop
        // inside it is a chain of dependent loads, so several blocks per SM
        // are needed to hide that latency.
        return multiprocessors * 8;
    }();
    return value;
}
}  // namespace

int decode_attention_min_segments(int query_heads) {
    if (query_heads <= 0) return 1;
    return std::max(1, (decode_attention_blocks_target() + query_heads - 1) / query_heads);
}

int decode_attention_segments(int query_heads, int max_context) {
    if (query_heads <= 0 || max_context <= 0) return 1;
    // Grid width and buffer size only: how many segments are *used* is decided
    // per step by decode_segment_plan from the live sequence length. This has
    // to cover the widest split that plan can ask for, which is a full context
    // divided kDecodeTokensPerSegment tokens at a time. Segments past the live
    // range exit immediately and cost nothing (measured at 13.0us for 6144
    // blocks against 13.1us for 768).
    const int context_bound =
        (max_context + kDecodeTokensPerSegment - 1) / kDecodeTokensPerSegment;
    return std::max({1, decode_attention_min_segments(query_heads), context_bound});
}

void launch_gqa_decode_segmented_device(const GqaDecodeSegmentedArgs& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const AttentionDecodeSegmentation& seg = args.segmentation;
    gqa_decode_segment_partial_kernel<<<args.geometry.q_heads * seg.segments, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        seg.segments, seg.min_segments, args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
    gqa_decode_segment_reduce_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.out, args.extent.position, args.geometry.q_heads,
        args.geometry.head_dim, seg.segments, seg.min_segments,
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

void launch_gqa_decode_segmented_int8_device(const GqaDecodeSegmentedInt8Args& args) {
    const int threads = attention_threads(args.geometry.head_dim);
    const AttentionDecodeSegmentation& seg = args.segmentation;
    gqa_decode_segment_partial_int8_kernel<<<args.geometry.q_heads * seg.segments, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.extent.position,
        args.geometry.q_heads, args.geometry.kv_heads, args.geometry.head_dim,
        seg.segments, seg.min_segments, args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
    gqa_decode_segment_reduce_kernel<<<args.geometry.q_heads, threads, 0, args.stream>>>(
        args.out, args.extent.position, args.geometry.q_heads,
        args.geometry.head_dim, seg.segments, seg.min_segments,
        args.geometry.sliding_window,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
