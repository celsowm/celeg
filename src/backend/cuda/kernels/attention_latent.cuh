
constexpr int kMaxLatentValuesPerLane = 16;

__global__ void factorized_latent_query_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* expansion,
    __nv_bfloat16* output, int rows, int heads, int nope, int rope, int rank) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * heads * rank;
    if (element >= total) return;
    const int r = static_cast<int>(element % rank);
    const int head = static_cast<int>((element / rank) % heads);
    const int row = static_cast<int>(element / (static_cast<size_t>(heads) * rank));
    const int q_stride = heads * (nope + rope);
    float sum = 0.0f;
    for (int d = 0; d < nope; ++d) {
        sum += __bfloat162float(query[static_cast<size_t>(row) * q_stride + head * (nope + rope) + d]) *
               __bfloat162float(expansion[static_cast<size_t>(head * nope + d) * rank + r]);
    }
    output[element] = __float2bfloat16(sum);
}

__global__ void factorized_latent_value_kernel(
    const __nv_bfloat16* latent, const __nv_bfloat16* expansion,
    __nv_bfloat16* output, int rows, int heads, int nope, int value_dim, int rank) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * heads * value_dim;
    if (element >= total) return;
    const int v = static_cast<int>(element % value_dim);
    const int head = static_cast<int>((element / value_dim) % heads);
    const int row = static_cast<int>(element / (static_cast<size_t>(heads) * value_dim));
    const int stride = nope + value_dim;
    float sum = 0.0f;
    for (int r = 0; r < rank; ++r) {
        sum += __bfloat162float(latent[static_cast<size_t>(row) * heads * rank + head * rank + r]) *
               __bfloat162float(expansion[static_cast<size_t>(head * stride + nope + v) * rank + r]);
    }
    output[element] = __float2bfloat16(sum);
}

__global__ void factorized_latent_rope_kernel(
    const __nv_bfloat16* query, __nv_bfloat16* output, int rows, int heads,
    int nope, int rope) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * heads * rope;
    if (element >= total) return;
    const int d = static_cast<int>(element % rope);
    const int head = static_cast<int>((element / rope) % heads);
    const int row = static_cast<int>(element / (static_cast<size_t>(heads) * rope));
    output[element] = query[static_cast<size_t>(row) * heads * (nope + rope) +
                            head * (nope + rope) + nope + d];
}

void launch_factorized_latent_query(const FactorizedLatentQueryArgs& args) {
    const size_t total =
        static_cast<size_t>(args.rows) * args.query_heads * args.latent_rank;
    factorized_latent_query_kernel<<<(total + 255) / 256, 256, 0, args.stream>>>(
        args.query_projection, args.expansion, args.query_content, args.rows,
        args.query_heads, args.query_nope, args.query_rope_dim,
        args.latent_rank);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_factorized_latent_value(const FactorizedLatentValueArgs& args) {
    const size_t total =
        static_cast<size_t>(args.rows) * args.query_heads * args.value_dim;
    factorized_latent_value_kernel<<<(total + 255) / 256, 256, 0, args.stream>>>(
        args.latent_output, args.expansion, args.value_output, args.rows,
        args.query_heads, args.query_nope, args.value_dim, args.latent_rank);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_factorized_latent_rope(const FactorizedLatentRopeArgs& args) {
    const size_t total =
        static_cast<size_t>(args.rows) * args.query_heads * args.query_rope_dim;
    factorized_latent_rope_kernel<<<(total + 255) / 256, 256, 0, args.stream>>>(
        args.query_projection, args.query_rope, args.rows, args.query_heads,
        args.query_nope, args.query_rope_dim);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

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

void launch_latent_attention_device(const LatentContiguousArgs& args) {
    const LatentGeometry& g = args.geometry;
    latent_attention_contiguous_kernel<<<g.query_heads, 32, 0, args.stream>>>(
        args.query.content, args.query.rope, args.kv.keys, args.kv.values,
        args.kv.key_rope, args.out, args.extent.position, args.alibi_slopes, 1,
        false, g.query_heads, g.latent_rank, g.rotary_width, g.score_scale,
        g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_latent_attention_prefill(const LatentContiguousArgs& args) {
    const LatentGeometry& g = args.geometry;
    const int rows = args.extent.rows;
    latent_attention_contiguous_kernel<<<rows * g.query_heads, 32, 0, args.stream>>>(
        args.query.content, args.query.rope, args.kv.keys, args.kv.values,
        args.kv.key_rope, args.out, nullptr, args.alibi_slopes, rows, true,
        g.query_heads, g.latent_rank, g.rotary_width, g.score_scale,
        g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_latent_attention_paged_batch(const LatentPagedArgs& args) {
    const LatentGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    latent_attention_paged_kernel<<<args.rows * g.query_heads, 32, 0, args.stream>>>(
        args.query.content, args.query.rope, args.kv.keys, args.kv.values,
        args.out, index.page_tables, index.page_table_stride, args.positions,
        args.alibi_slopes, args.rows, index.attention_slot, index.page_tokens,
        index.page_vector_elements, index.layer_vector_offset, g.query_heads,
        g.latent_rank, g.rotary_width, g.score_scale, g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_latent_attention_batch_ptrs(const LatentBatchPtrArgs& args) {
    const LatentGeometry& g = args.geometry;
    latent_attention_ptr_kernel<<<args.rows * g.query_heads, 32, 0, args.stream>>>(
        args.query.content, args.query.rope, args.kv.keys, args.kv.values,
        args.kv.key_rope, args.out, args.positions, args.alibi_slopes,
        args.rows, g.query_heads, g.latent_rank, g.rotary_width, g.score_scale,
        g.sliding_window);
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
