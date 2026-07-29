// Multi-request batched decode against a paged KV cache: one shared pool of
// fixed-size pages plus a per-session page table, so sessions grow without
// reserving a contiguous max-context slab each.
//
// Every KV access goes through paged_vector_offset / paged_scale_offset from
// kv_cache.inl to translate a logical token index into (page, slot) -- which is
// why this leaf must be included after kv_cache.inl. It is the only attention
// leaf with that dependency.
//
// Strict/online are selected by a template parameter rather than the `mode`
// runtime switch used elsewhere, plus a segmented partial/reduce pair for long
// contexts.

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
