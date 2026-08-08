// Reference CUDA lowering for latent-key attention.  Persistent state is
// [latent key/value rank] plus an optional decoupled rotary key per token.
// The implementation deliberately uses one warp per query head so it can be
// used as a correctness path before a fused decompression kernel is added.

constexpr int kMaxLatentValuesPerLane = 16;

__device__ __forceinline__ float latent_score(
    float dot, const float* slopes, int head, int query_position,
    int key_position, float scale) {
    if (slopes) {
        dot -= slopes[head] * static_cast<float>(query_position - key_position);
    }
    return dot * scale;
}

__device__ __forceinline__ void latent_online_update(
    float score, float& running_max, float& denominator, float* accumulator,
    const __nv_bfloat16* value, int latent_rank, int lane) {
    const float next_max = fmaxf(running_max, score);
    const float alpha = expf(running_max - next_max);
    const float beta = expf(score - next_max);
    denominator = denominator * alpha + beta;
    int index = 0;
    for (int d = lane; d < latent_rank; d += 32, ++index) {
        accumulator[index] = accumulator[index] * alpha +
            bf16_float(value[d]) * beta;
    }
    running_max = next_max;
}

__global__ void latent_attention_contiguous_kernel(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache,
    const __nv_bfloat16* key_rope_cache, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int rows, bool prefill,
    int query_heads, int latent_rank, int rotary_width, float score_scale,
    int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / query_heads;
    const int head = flat % query_heads;
    if (row >= rows) return;
    const int query_position = prefill ? row : *position;
    const int sequence_length = query_position + 1;
    const int first = sliding_window > 0
        ? max(0, sequence_length - sliding_window) : 0;
    const int lane = threadIdx.x;
    const __nv_bfloat16* query = query_content +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    const __nv_bfloat16* query_r = query_rope
        ? query_rope + (static_cast<size_t>(row) * query_heads + head) * rotary_width
        : nullptr;
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxLatentValuesPerLane];
#pragma unroll
    for (int i = 0; i < kMaxLatentValuesPerLane; ++i) accumulator[i] = 0.0f;
    for (int token = first; token < sequence_length; ++token) {
        const __nv_bfloat16* key = key_cache +
            static_cast<size_t>(token) * latent_rank;
        const __nv_bfloat16* key_r = key_rope_cache && rotary_width
            ? key_rope_cache + static_cast<size_t>(token) * rotary_width : nullptr;
        float partial = 0.0f;
        for (int d = lane; d < latent_rank; d += 32) {
            partial += bf16_float(query[d]) * bf16_float(key[d]);
        }
        if (query_r && key_r) {
            for (int d = lane; d < rotary_width; d += 32) {
                partial += bf16_float(query_r[d]) * bf16_float(key_r[d]);
            }
        }
        const float score = latent_score(warp_broadcast_sum(partial), slopes,
            head, query_position, token, score_scale);
        latent_online_update(score, running_max, denominator, accumulator,
                             value_cache + static_cast<size_t>(token) * latent_rank,
                             latent_rank, lane);
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    int index = 0;
    for (int d = lane; d < latent_rank; d += 32, ++index) {
        output[d] = __float2bfloat16(accumulator[index] / denominator);
    }
}

__global__ void latent_attention_paged_kernel(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    __nv_bfloat16* out, const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, const float* slopes, int rows, int slot,
    int page_tokens, size_t page_elements, size_t layer_offset,
    int query_heads, int latent_rank, int rotary_width, float score_scale,
    int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / query_heads;
    const int head = flat % query_heads;
    if (row >= rows) return;
    const int query_position = positions[row];
    const int sequence_length = query_position + 1;
    const int first = sliding_window > 0
        ? max(0, sequence_length - sliding_window) : 0;
    const int lane = threadIdx.x;
    const __nv_bfloat16* query = query_content +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    const __nv_bfloat16* query_r = query_rope
        ? query_rope + (static_cast<size_t>(row) * query_heads + head) * rotary_width
        : nullptr;
    const size_t token_width = static_cast<size_t>(latent_rank + rotary_width);
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator[kMaxLatentValuesPerLane];
#pragma unroll
    for (int i = 0; i < kMaxLatentValuesPerLane; ++i) accumulator[i] = 0.0f;
    for (int token = first; token < sequence_length; ++token) {
        const uint32_t page = page_tables[
            static_cast<size_t>(row) * page_table_stride + token / page_tokens];
        const size_t base = static_cast<size_t>(page) * page_elements +
            layer_offset + static_cast<size_t>(token % page_tokens) * token_width;
        float partial = 0.0f;
        for (int d = lane; d < latent_rank; d += 32) {
            partial += bf16_float(query[d]) * bf16_float(key_pool[base + d]);
        }
        if (query_r && rotary_width) {
            for (int d = lane; d < rotary_width; d += 32) {
                partial += bf16_float(query_r[d]) *
                    bf16_float(key_pool[base + latent_rank + d]);
            }
        }
        const float score = latent_score(warp_broadcast_sum(partial), slopes,
            head, query_position, token, score_scale);
        latent_online_update(score, running_max, denominator, accumulator,
                             value_pool + base, latent_rank, lane);
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    int index = 0;
    for (int d = lane; d < latent_rank; d += 32, ++index) {
        output[d] = __float2bfloat16(accumulator[index] / denominator);
    }
}

__global__ void latent_attention_ptr_kernel(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* const* keys, const __nv_bfloat16* const* values,
    const __nv_bfloat16* const* key_ropes, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int query_heads,
    int latent_rank, int rotary_width, float score_scale, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / query_heads;
    const int head = flat % query_heads;
    if (row >= rows) return;
    const int query_position = positions[row];
    const int sequence_length = query_position + 1;
    const int first = sliding_window > 0
        ? max(0, sequence_length - sliding_window) : 0;
    const int lane = threadIdx.x;
    const __nv_bfloat16* query = query_content +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    const __nv_bfloat16* query_r = query_rope
        ? query_rope + (static_cast<size_t>(row) * query_heads + head) * rotary_width
        : nullptr;
    const __nv_bfloat16* row_keys = keys[row];
    const __nv_bfloat16* row_values = values[row];
    const __nv_bfloat16* row_key_rope = key_ropes ? key_ropes[row] : nullptr;
    float running_max = -FLT_MAX, denominator = 0.0f;
    float accumulator[kMaxLatentValuesPerLane];
#pragma unroll
    for (int i = 0; i < kMaxLatentValuesPerLane; ++i) accumulator[i] = 0.0f;
    for (int token = first; token < sequence_length; ++token) {
        const __nv_bfloat16* key = row_keys + static_cast<size_t>(token) * latent_rank;
        const __nv_bfloat16* key_r = row_key_rope && rotary_width
            ? row_key_rope + static_cast<size_t>(token) * rotary_width : nullptr;
        float partial = 0.0f;
        for (int d = lane; d < latent_rank; d += 32)
            partial += bf16_float(query[d]) * bf16_float(key[d]);
        if (query_r && key_r) {
            for (int d = lane; d < rotary_width; d += 32)
                partial += bf16_float(query_r[d]) * bf16_float(key_r[d]);
        }
        const float score = latent_score(warp_broadcast_sum(partial), slopes,
            head, query_position, token, score_scale);
        latent_online_update(score, running_max, denominator, accumulator,
                             row_values + static_cast<size_t>(token) * latent_rank,
                             latent_rank, lane);
    }
    __nv_bfloat16* output = out +
        (static_cast<size_t>(row) * query_heads + head) * latent_rank;
    int index = 0;
    for (int d = lane; d < latent_rank; d += 32, ++index)
        output[d] = __float2bfloat16(accumulator[index] / denominator);
}

void launch_latent_attention_device(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache,
    const __nv_bfloat16* key_rope_cache, __nv_bfloat16* out,
    const int32_t* position, const float* slopes, int query_heads,
    int latent_rank, int rotary_width, float score_scale,
    int sliding_window, cudaStream_t stream) {
    latent_attention_contiguous_kernel<<<query_heads, 32, 0, stream>>>(
        query_content, query_rope, key_cache, value_cache, key_rope_cache, out,
        position, slopes, 1, false, query_heads, latent_rank, rotary_width,
        score_scale, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_latent_attention_prefill(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* key_cache, const __nv_bfloat16* value_cache,
    const __nv_bfloat16* key_rope_cache, __nv_bfloat16* out, int rows,
    const float* slopes, int query_heads, int latent_rank, int rotary_width,
    float score_scale, int sliding_window, cudaStream_t stream) {
    latent_attention_contiguous_kernel<<<rows * query_heads, 32, 0, stream>>>(
        query_content, query_rope, key_cache, value_cache, key_rope_cache, out,
        nullptr, slopes, rows, true, query_heads, latent_rank, rotary_width,
        score_scale, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_latent_attention_paged_batch(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    __nv_bfloat16* out, const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, const float* slopes, int rows, int slot,
    int page_tokens, size_t page_elements, size_t layer_offset,
    int query_heads, int latent_rank, int rotary_width, float score_scale,
    int sliding_window, cudaStream_t stream) {
    latent_attention_paged_kernel<<<rows * query_heads, 32, 0, stream>>>(
        query_content, query_rope, key_pool, value_pool, out, page_tables,
        page_table_stride, positions, slopes, rows, slot, page_tokens,
        page_elements, layer_offset, query_heads, latent_rank, rotary_width,
        score_scale, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_latent_attention_batch_ptrs(
    const __nv_bfloat16* query_content, const __nv_bfloat16* query_rope,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    const __nv_bfloat16* const* key_rope_cache, __nv_bfloat16* out,
    const int32_t* positions, const float* slopes, int rows, int query_heads,
    int latent_rank, int rotary_width, float score_scale, int sliding_window,
    cudaStream_t stream) {
    latent_attention_ptr_kernel<<<rows * query_heads, 32, 0, stream>>>(
        query_content, query_rope, key_cache, value_cache, key_rope_cache, out,
        positions, slopes, rows, query_heads, latent_rank, rotary_width,
        score_scale, sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
