
__device__ __forceinline__ int relative_position_bucket(
    int query_position, int key_position, int total_bucket_count,
    int max_distance, bool bidirectional) {
    const int relative_position = key_position - query_position;
    const int bucket_count = bidirectional
        ? total_bucket_count / 2 : total_bucket_count;
    const bool positive = bidirectional && relative_position > 0;
    const int distance = bidirectional
        ? abs(relative_position) : max(-relative_position, 0);
    const int max_exact = bucket_count / 2;
    int bucket = 0;
    if (distance < max_exact) {
        bucket = distance;
    } else {
        const int safe_exact = max(max_exact, 1);
        const int safe_distance = max(distance, max_exact);
        const int safe_max_distance = max(max_distance, max_exact + 1);
        const float denominator = logf(
            static_cast<float>(safe_max_distance) /
            static_cast<float>(safe_exact));
        const float logarithmic = denominator == 0.0f ? 0.0f : logf(
            static_cast<float>(safe_distance) /
            static_cast<float>(safe_exact)) / denominator;
        bucket = max_exact + static_cast<int>(
            logarithmic * static_cast<float>(bucket_count - max_exact));
        bucket = min(bucket, bucket_count - 1);
    }
    if (positive) bucket += bucket_count;
    return bucket;
}

__device__ __forceinline__ float relative_bias_score(
    float dot, float scale, const float* values, int total_bucket_count,
    int max_distance, bool bidirectional, int head,
    int query_position, int key_position) {
    const int bucket = relative_position_bucket(
        query_position, key_position, total_bucket_count,
        max_distance, bidirectional);
    return dot * scale + values[static_cast<size_t>(head) *
        static_cast<size_t>(total_bucket_count) + static_cast<size_t>(bucket)];
}

__global__ void gqa_relative_contiguous_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, bool prefill,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = prefill ? row : *position;
    const int seq_len = query_position + 1;
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
        const size_t base = (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            partial += bf16_float(query[d]) * bf16_float(key_cache[base + d]);
        }
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        const __nv_bfloat16* value = value_cache + base;
        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index) {
            accumulator[index] = accumulator[index] * alpha +
                bf16_float(value[d]) * beta;
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

__global__ void gqa_relative_contiguous_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache, const int8_t* value_cache,
    const float* key_scales, const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, bool prefill,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = prefill ? row : *position;
    const int seq_len = query_position + 1;
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
        const int8_t* key = key_cache + scale_index * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            partial += bf16_float(query[d]) * static_cast<float>(key[d]) *
                key_scales[scale_index];
        }
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        const int8_t* value = value_cache + scale_index * head_dim;
        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index) {
            accumulator[index] = accumulator[index] * alpha +
                static_cast<float>(value[d]) * value_scales[scale_index] * beta;
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

__global__ void gqa_relative_ptr_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* const* keys,
    const __nv_bfloat16* const* values, __nv_bfloat16* out,
    const int32_t* positions, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, int q_heads,
    int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = positions[row];
    const int seq_len = query_position + 1;
    const int first = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    const __nv_bfloat16* row_keys = keys[row];
    const __nv_bfloat16* row_values = values[row];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX, denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const int lane = threadIdx.x;
    for (int token = first; token < seq_len; ++token) {
        const size_t base = (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32)
            partial += bf16_float(query[d]) * bf16_float(row_keys[base + d]);
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index)
            accumulator[index] = accumulator[index] * alpha +
                bf16_float(row_values[base + d]) * beta;
        running_max = next_max;
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    int index = 0;
    for (int d = lane; d < head_dim; d += 32, ++index)
        output[d] = __float2bfloat16(accumulator[index] / denominator);
}

__global__ void gqa_relative_int8_ptr_kernel(
    const __nv_bfloat16* q, const int8_t* const* keys,
    const int8_t* const* values, const float* const* key_scales,
    const float* const* value_scales, __nv_bfloat16* out,
    const int32_t* positions, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, int q_heads,
    int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = positions[row];
    const int seq_len = query_position + 1;
    const int first = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    const int8_t* row_keys = keys[row];
    const int8_t* row_values = values[row];
    const float* row_key_scales = key_scales[row];
    const float* row_value_scales = value_scales[row];
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX, denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    const int lane = threadIdx.x;
    for (int token = first; token < seq_len; ++token) {
        const size_t si = static_cast<size_t>(token) * kv_heads + kv_head;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32)
            partial += bf16_float(query[d]) *
                static_cast<float>(row_keys[si * head_dim + d]) * row_key_scales[si];
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index)
            accumulator[index] = accumulator[index] * alpha +
                static_cast<float>(row_values[si * head_dim + d]) * row_value_scales[si] * beta;
        running_max = next_max;
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    int index = 0;
    for (int d = lane; d < head_dim; d += 32, ++index)
        output[d] = __float2bfloat16(accumulator[index] / denominator);
}

__global__ void gqa_relative_paged_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, const uint32_t* tables, int stride,
    __nv_bfloat16* out, const int32_t* positions, const float* bias,
    int bucket_count, int max_distance, bool bidirectional,
    int rows, int slot, int page_tokens, size_t page_elements,
    size_t layer_offset, int q_heads, int kv_heads, int head_dim,
    int sliding_window) {
    const int flat = blockIdx.x, row = flat / q_heads, head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = positions[row], seq_len = query_position + 1;
    const int first = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads), lane = threadIdx.x;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX, denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    for (int token = first; token < seq_len; ++token) {
        const uint32_t page = tables[static_cast<size_t>(row) * stride + token / page_tokens];
        const int in_page = token % page_tokens;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            const size_t offset = paged_vector_offset(page, slot, in_page, kv_head, d,
                page_tokens, page_elements, layer_offset, kv_heads, head_dim);
            partial += bf16_float(query[d]) * bf16_float(key_pool[offset]);
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
            const size_t offset = paged_vector_offset(page, slot, in_page, kv_head, d,
                page_tokens, page_elements, layer_offset, kv_heads, head_dim);
            accumulator[index] = accumulator[index] * alpha +
                bf16_float(value_pool[offset]) * beta;
        }
        running_max = next_max;
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    int index = 0;
    for (int d = lane; d < head_dim; d += 32, ++index)
        output[d] = __float2bfloat16(accumulator[index] / denominator);
}

__global__ void gqa_relative_paged_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scales, const float* value_scales,
    const uint32_t* tables, int stride, __nv_bfloat16* out,
    const int32_t* positions, const float* bias, int bucket_count,
    int max_distance, bool bidirectional, int rows, int slot,
    int page_tokens, size_t page_elements, size_t layer_offset,
    size_t page_scale_elements, size_t layer_scale_offset, int q_heads,
    int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x, row = flat / q_heads, head = flat % q_heads;
    if (row >= rows) return;
    const int query_position = positions[row], seq_len = query_position + 1;
    const int first = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int kv_head = head / (q_heads / kv_heads), lane = threadIdx.x;
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    float running_max = -FLT_MAX, denominator = 0.0f;
    float accumulator[kMaxHeadDimPerLane];
#pragma unroll
    for (int i = 0; i < kMaxHeadDimPerLane; ++i) accumulator[i] = 0.0f;
    for (int token = first; token < seq_len; ++token) {
        const uint32_t page = tables[static_cast<size_t>(row) * stride + token / page_tokens];
        const int in_page = token % page_tokens;
        const size_t so = paged_scale_offset(page, slot, in_page, kv_head,
            page_tokens, page_scale_elements, layer_scale_offset, kv_heads);
        const float key_scale = key_scales[so];
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            const size_t offset = paged_vector_offset(page, slot, in_page, kv_head, d,
                page_tokens, page_elements, layer_offset, kv_heads, head_dim);
            partial += bf16_float(query[d]) * static_cast<float>(key_pool[offset]) * key_scale;
        }
        const float score = relative_bias_score(
            warp_broadcast_sum(partial), scale, bias, bucket_count,
            max_distance, bidirectional, head, query_position, token);
        const float next_max = fmaxf(running_max, score);
        const float alpha = expf(running_max - next_max);
        const float beta = expf(score - next_max);
        denominator = denominator * alpha + beta;
        const float value_scale = value_scales[so];
        int index = 0;
        for (int d = lane; d < head_dim; d += 32, ++index) {
            const size_t offset = paged_vector_offset(page, slot, in_page, kv_head, d,
                page_tokens, page_elements, layer_offset, kv_heads, head_dim);
            accumulator[index] = accumulator[index] * alpha +
                static_cast<float>(value_pool[offset]) * value_scale * beta;
        }
        running_max = next_max;
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * q_heads + head) * head_dim;
    int index = 0;
    for (int d = lane; d < head_dim; d += 32, ++index)
        output[d] = __float2bfloat16(accumulator[index] / denominator);
}

void launch_gqa_decode_relative_device(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    gqa_relative_contiguous_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out,
        args.extent.position, bias.values, bias.bucket_count,
        bias.max_distance, bias.bidirectional, 1, false,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_relative(const GqaContiguousArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    gqa_relative_contiguous_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, nullptr,
        bias.values, bias.bucket_count, bias.max_distance, bias.bidirectional,
        rows, true, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_device(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    gqa_relative_contiguous_int8_kernel<<<g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, args.extent.position, bias.values,
        bias.bucket_count, bias.max_distance, bias.bidirectional, 1, false,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_prefill_relative_int8(const GqaContiguousInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    const int rows = args.extent.rows;
    gqa_relative_contiguous_int8_kernel<<<rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, nullptr, bias.values,
        bias.bucket_count, bias.max_distance, bias.bidirectional, rows, true,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_batch_ptrs(const GqaBatchPtrArgs& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    gqa_relative_ptr_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, args.positions,
        bias.values, bias.bucket_count, bias.max_distance, bias.bidirectional,
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_batch_ptrs(const GqaBatchPtrInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const auto& bias = args.relative_bias;
    gqa_relative_int8_ptr_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, args.out, args.positions, bias.values,
        bias.bucket_count, bias.max_distance, bias.bidirectional,
        args.rows, g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_paged_batch(const GqaPagedArgs& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const auto& bias = args.relative_bias;
    gqa_relative_paged_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, index.page_tables,
        index.page_table_stride, args.out, args.positions, bias.values,
        bias.bucket_count, bias.max_distance, bias.bidirectional,
        args.rows, index.attention_slot, index.page_tokens,
        index.page_vector_elements, index.layer_vector_offset,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_relative_int8_paged_batch(const GqaPagedInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const PagedKvScaleIndex& scales = args.scale_index;
    const auto& bias = args.relative_bias;
    gqa_relative_paged_int8_kernel<<<args.rows * g.q_heads, 32, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, index.page_tables, index.page_table_stride,
        args.out, args.positions, bias.values, bias.bucket_count,
        bias.max_distance, bias.bidirectional, args.rows,
        index.attention_slot, index.page_tokens, index.page_vector_elements,
        index.layer_vector_offset, scales.page_scale_elements,
        scales.layer_scale_offset,
        g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
