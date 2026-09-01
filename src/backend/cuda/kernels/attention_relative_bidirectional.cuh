__device__ __forceinline__ float relative_bidirectional_cache_value(
    const __nv_bfloat16* values, size_t index, const float*, size_t) {
    return bf16_float(values[index]);
}

__device__ __forceinline__ float relative_bidirectional_cache_value(
    const int8_t* values, size_t index, const float* scales, size_t scale_index) {
    return static_cast<float>(values[index]) * scales[scale_index];
}

template <typename CacheT>
__global__ void gqa_relative_bidirectional_prefill_kernel(
    const __nv_bfloat16* q, const CacheT* key_cache, const CacheT* value_cache,
    const float* key_scales, const float* value_scales,
    __nv_bfloat16* out, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, int seq_len,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;

    const int query_position = row;
    const int first = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;

    const int lane = threadIdx.x;
    for (int token = first; token < seq_len; ++token) {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const size_t base = scale_index * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            partial += bf16_float(query[d]) * relative_bidirectional_cache_value(
                key_cache, base + static_cast<size_t>(d), key_scales, scale_index);
        }
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;

        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index) {
            accumulator[index] = accumulator[index] * alpha +
                relative_bidirectional_cache_value(
                    value_cache, base + static_cast<size_t>(d),
                    value_scales, scale_index) * beta;
        }
        running_max = next_max;
    }

    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    int index = 0;
    for (int d = lane; d < head_dim; d += 32, ++index) {
        output[d] = __float2bfloat16(accumulator[index] / denominator);
    }
}

void launch_gqa_prefill_relative_bidirectional(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    gqa_relative_bidirectional_prefill_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values,
        nullptr, nullptr, args.out, bias.values, bias.bucket_count,
        bias.max_distance, bias.bidirectional, rows, rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_relative_bidirectional_int8(
    const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    gqa_relative_bidirectional_prefill_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values,
        args.kv.key_scales, args.kv.value_scales,
        args.out, bias.values, bias.bucket_count,
        bias.max_distance, bias.bidirectional, rows, rows,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
