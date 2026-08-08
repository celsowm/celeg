// ALiBi lowering.  This is intentionally a small, explicit online-softmax
// implementation: ALiBi changes every score, so routing it through the
// bias-free kernels would silently produce the wrong model.  The ordinary
// kernels remain on their existing fast paths.

__device__ __forceinline__ float alibi_score(float dot, float scale,
                                             const float* slopes,
                                             int head, int query_position,
                                             int key_position) {
    return dot * scale - slopes[head] *
        static_cast<float>(query_position - key_position);
}

__global__ void gqa_alibi_contiguous_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int rows, bool prefill,
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
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

__global__ void gqa_alibi_contiguous_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_cache, const int8_t* value_cache,
    const float* key_scales, const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int rows, bool prefill,
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
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

__global__ void gqa_alibi_ptr_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* const* keys,
    const __nv_bfloat16* const* values, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int q_heads,
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
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

__global__ void gqa_alibi_int8_ptr_kernel(
    const __nv_bfloat16* q, const int8_t* const* keys,
    const int8_t* const* values, const float* const* key_scales,
    const float* const* value_scales, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int q_heads,
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
            partial += bf16_float(query[d]) * static_cast<float>(row_keys[si * head_dim + d]) * row_key_scales[si];
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

__global__ void gqa_alibi_paged_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, const uint32_t* tables, int stride,
    __nv_bfloat16* out, const int32_t* positions, const float* slopes,
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
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

__global__ void gqa_alibi_paged_int8_kernel(
    const __nv_bfloat16* q, const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scales, const float* value_scales,
    const uint32_t* tables, int stride, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int slot,
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
        const float score = alibi_score(warp_broadcast_sum(partial), scale,
                                        slopes, head, query_position, token);
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

void launch_gqa_decode_alibi_device(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int q_heads, int kv_heads,
    int head_dim, int sliding_window, cudaStream_t stream) {
    gqa_alibi_contiguous_kernel<<<q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, out, position, slopes, 1, false,
        q_heads, kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_alibi(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out, int rows,
    const float* slopes, int q_heads, int kv_heads, int head_dim,
    int sliding_window, cudaStream_t stream) {
    gqa_alibi_contiguous_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, out, nullptr, slopes, rows, true,
        q_heads, kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_alibi_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache, const int8_t* value_cache,
    const float* key_scales, const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int q_heads, int kv_heads,
    int head_dim, int sliding_window, cudaStream_t stream) {
    gqa_alibi_contiguous_int8_kernel<<<q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, position,
        slopes, 1, false, q_heads, kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_prefill_alibi_int8(
    const __nv_bfloat16* q, const int8_t* key_cache, const int8_t* value_cache,
    const float* key_scales, const float* value_scales, __nv_bfloat16* out,
    int rows, const float* slopes, int q_heads, int kv_heads, int head_dim,
    int sliding_window, cudaStream_t stream) {
    gqa_alibi_contiguous_int8_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_cache, value_cache, key_scales, value_scales, out, nullptr,
        slopes, rows, true, q_heads, kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_alibi_batch_ptrs(
    const __nv_bfloat16* q, const __nv_bfloat16* const* keys,
    const __nv_bfloat16* const* values, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int q_heads,
    int kv_heads, int head_dim, int sliding_window, cudaStream_t stream) {
    gqa_alibi_ptr_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, keys, values, out, positions, slopes, rows, q_heads, kv_heads,
        head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_alibi_int8_batch_ptrs(
    const __nv_bfloat16* q, const int8_t* const* keys,
    const int8_t* const* values, const float* const* key_scales,
    const float* const* value_scales, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int q_heads,
    int kv_heads, int head_dim, int sliding_window, cudaStream_t stream) {
    gqa_alibi_int8_ptr_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, keys, values, key_scales, value_scales, out, positions, slopes,
        rows, q_heads, kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_alibi_paged_batch(
    const __nv_bfloat16* q, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, const uint32_t* tables, int stride,
    __nv_bfloat16* out, const int32_t* positions, const float* slopes,
    int rows, int slot, int page_tokens, size_t page_elements,
    size_t layer_offset, int q_heads, int kv_heads, int head_dim,
    int sliding_window, cudaStream_t stream) {
    gqa_alibi_paged_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_pool, value_pool, tables, stride, out, positions, slopes, rows,
        slot, page_tokens, page_elements, layer_offset, q_heads, kv_heads,
        head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_gqa_decode_alibi_int8_paged_batch(
    const __nv_bfloat16* q, const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scales, const float* value_scales,
    const uint32_t* tables, int stride, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int slot,
    int page_tokens, size_t page_elements, size_t layer_offset,
    size_t page_scale_elements, size_t layer_scale_offset, int q_heads,
    int kv_heads, int head_dim, int sliding_window, cudaStream_t stream) {
    gqa_alibi_paged_int8_kernel<<<rows * q_heads, 32, 0, stream>>>(
        q, key_pool, value_pool, key_scales, value_scales, tables, stride, out,
        positions, slopes, rows, slot, page_tokens, page_elements,
        layer_offset, page_scale_elements, layer_scale_offset, q_heads,
        kv_heads, head_dim, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
