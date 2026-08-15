
template <bool Strict>
__global__ void gqa_decode_paged_batch_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, size_t page_vector_elements,
    size_t layer_vector_offset,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens;
            const int ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            float local = 0.0f;
            for (int d = lane; d < head_dim; d += blockDim.x) {
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, d, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    size_t page_scale_elements, size_t layer_scale_offset,
    int kv_heads, int head_dim,
    float* warp_sums, float* dot_total) {
    const int lane = threadIdx.x;
    const size_t scale_offset = paged_scale_offset(
        page, attention_slot, in_page, kv_head, page_tokens,
        page_scale_elements, layer_scale_offset, kv_heads);
    const float key_scale = key_scale_pool[scale_offset];
    float local = 0.0f;
    for (int d = lane; d < head_dim; d += blockDim.x) {
        const size_t offset = paged_vector_offset(
            page, attention_slot, in_page, kv_head, d, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
    int attention_slot, int page_tokens, size_t page_vector_elements,
    size_t layer_vector_offset, size_t page_scale_elements,
    size_t layer_scale_offset,
    int q_heads, int kv_heads, int head_dim, int sliding_window) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows) return;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, page_vector_elements, layer_vector_offset,
                page_scale_elements, layer_scale_offset, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) maximum = fmaxf(maximum,
                rounded_bf16_float(rounded_bf16_float(dot) * scale));
            __syncthreads();
        }
        if (lane == 0) denominator = 0.0f;
        __syncthreads();
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, page_vector_elements, layer_vector_offset,
                page_scale_elements, layer_scale_offset, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) denominator += expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum);
            __syncthreads();
        }
        float accumulator = 0.0f;
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, page_vector_elements, layer_vector_offset,
                page_scale_elements, layer_scale_offset, kv_heads, head_dim,
                warp_sums, &dot_total);
            if (lane == 0) probability = rounded_bf16_float(expf(
                rounded_bf16_float(rounded_bf16_float(dot) * scale) - maximum) / denominator);
            __syncthreads();
            if (lane < head_dim) {
                const size_t scale_offset = paged_scale_offset(
                    page, attention_slot, ip, kv_head, page_tokens,
                    page_scale_elements, layer_scale_offset, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
        for (int token = first_token; token < seq_len; ++token) {
            const int lp = token / page_tokens, ip = token % page_tokens;
            const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + lp];
            const float dot = paged_int8_attention_dot(
                query, key_pool, key_scale_pool, page, attention_slot, ip,
                kv_head, page_tokens, page_vector_elements, layer_vector_offset,
                page_scale_elements, layer_scale_offset, kv_heads, head_dim,
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
                    page_scale_elements, layer_scale_offset, kv_heads);
                const size_t offset = paged_vector_offset(
                    page, attention_slot, ip, kv_head, lane, page_tokens,
                    page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    int q_heads, int kv_heads,
        int head_dim, int chunk_tokens, int chunks, int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int begin = max(chunk * chunk_tokens, first_token);
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
                page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
                page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    size_t page_scale_elements, size_t layer_scale_offset,
    int q_heads, int kv_heads,
        int head_dim, int chunk_tokens, int chunks, int sliding_window,
    float* partial_max, float* partial_denom, float* partial_accum) {
    const int flat = blockIdx.x;
    const int chunk = flat % chunks;
    const int query_index = flat / chunks;
    const int row = query_index / q_heads;
    const int query_head = query_index % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int begin = max(chunk * chunk_tokens, first_token);
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
            kv_head, page_tokens, page_vector_elements, layer_vector_offset,
            page_scale_elements, layer_scale_offset, kv_heads, head_dim,
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
                page_scale_elements, layer_scale_offset, kv_heads);
            const size_t offset = paged_vector_offset(
                page, attention_slot, ip, kv_head, lane, page_tokens,
                page_vector_elements, layer_vector_offset, kv_heads, head_dim);
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

void launch_gqa_decode_paged_batch(const GqaPagedArgs& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    if (args.fast) {
        gqa_decode_paged_batch_kernel<false><<<args.rows * g.q_heads, 64, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, index.page_tables,
            index.page_table_stride, args.out, args.positions, args.rows,
            index.attention_slot, index.page_tokens, index.page_vector_elements,
            index.layer_vector_offset,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_paged_batch_kernel<true><<<args.rows * g.q_heads, 64, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, index.page_tables,
            index.page_table_stride, args.out, args.positions, args.rows,
            index.attention_slot, index.page_tokens, index.page_vector_elements,
            index.layer_vector_offset,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_batch(const GqaPagedInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const PagedKvScaleIndex& scales = args.scale_index;
    if (args.fast) {
        gqa_decode_int8_paged_batch_kernel<false><<<args.rows * g.q_heads, 64, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
            args.kv.value_scales, index.page_tables, index.page_table_stride,
            args.out, args.positions, args.rows, index.attention_slot,
            index.page_tokens, index.page_vector_elements,
            index.layer_vector_offset, scales.page_scale_elements,
            scales.layer_scale_offset,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_int8_paged_batch_kernel<true><<<args.rows * g.q_heads, 64, 0, args.stream>>>(
            args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
            args.kv.value_scales, index.page_tables, index.page_table_stride,
            args.out, args.positions, args.rows, index.attention_slot,
            index.page_tokens, index.page_vector_elements,
            index.layer_vector_offset, scales.page_scale_elements,
            scales.layer_scale_offset,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_CHECK();
}

void launch_gqa_decode_paged_segmented_batch(const GqaPagedSegmentedArgs& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const AttentionSegmentation& seg = args.segmentation;
    const int threads = attention_threads(g.head_dim);
    gqa_decode_paged_segment_partial_kernel<<<args.rows * g.q_heads * seg.chunks, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, index.page_tables,
        index.page_table_stride, args.positions, args.rows,
        index.attention_slot, index.page_tokens, index.page_vector_elements,
        index.layer_vector_offset, g.q_heads, g.kv_heads, g.head_dim,
        seg.chunk_tokens, seg.chunks, g.sliding_window, seg.partial_max,
        seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
        args.out, args.rows, g.q_heads, g.head_dim, seg.chunks,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_CHECK();
}

void launch_gqa_decode_int8_paged_segmented_batch(const GqaPagedSegmentedInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const PagedKvIndex& index = args.index;
    const PagedKvScaleIndex& scales = args.scale_index;
    const AttentionSegmentation& seg = args.segmentation;
    const int threads = attention_threads(g.head_dim);
    gqa_decode_int8_paged_segment_partial_kernel<<<args.rows * g.q_heads * seg.chunks, threads, 0, args.stream>>>(
        args.query, args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, index.page_tables, index.page_table_stride,
        args.positions, args.rows, index.attention_slot, index.page_tokens,
        index.page_vector_elements, index.layer_vector_offset,
        scales.page_scale_elements, scales.layer_scale_offset,
        g.q_heads, g.kv_heads, g.head_dim, seg.chunk_tokens, seg.chunks,
        g.sliding_window, seg.partial_max, seg.partial_denom,
        seg.partial_accum);
    CELEG_KERNEL_CHECK();
    gqa_decode_segment_reduce_batch_kernel<<<args.rows * g.q_heads, threads, 0, args.stream>>>(
        args.out, args.rows, g.q_heads, g.head_dim, seg.chunks,
        seg.partial_max, seg.partial_denom, seg.partial_accum);
    CELEG_KERNEL_CHECK();
}
