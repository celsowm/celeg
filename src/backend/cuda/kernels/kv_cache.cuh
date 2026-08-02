__device__ __forceinline__ size_t paged_vector_offset(
    uint32_t page, int attention_slot, int in_page, int head, int dim,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    int kv_heads, int head_dim) {
    return static_cast<size_t>(page) * page_vector_elements + layer_vector_offset +
        ((static_cast<size_t>(in_page) * kv_heads + head) * head_dim) + dim;
}

__device__ __forceinline__ size_t paged_scale_offset(
    uint32_t page, int attention_slot, int in_page, int head,
    int page_tokens, size_t page_scale_elements, size_t layer_scale_offset,
    int kv_heads) {
    return static_cast<size_t>(page) * page_scale_elements + layer_scale_offset +
        static_cast<size_t>(in_page) * kv_heads + head;
}

__global__ void store_kv_kernel(const __nv_bfloat16* k,
                                const __nv_bfloat16* v,
                                __nv_bfloat16* key_cache,
                                __nv_bfloat16* value_cache,
                                int rows,
                                int width,
                                int position_value,
                                const int32_t* position_pointer,
                                int mode) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * width;
    if (index >= total) return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    key_cache[static_cast<size_t>(target_row) * width + column] = k[index];
    value_cache[static_cast<size_t>(target_row) * width + column] = v[index];
}

__global__ void store_kv_int8_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_cache, int8_t* value_cache,
    float* key_scales, float* value_scales, int rows, int kv_heads,
    int head_dim, int position_value, const int32_t* position_pointer,
    int mode) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows || head >= kv_heads) return;
    const int base_position = mode == 1 ? *position_pointer : position_value;
    const int target_row = mode == 2 ? row : base_position + row;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t cache_base =
        (static_cast<size_t>(target_row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index =
            static_cast<size_t>(target_row) * kv_heads + head;
        key_scales[scale_index] = key_scale;
        value_scales[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

__global__ void store_kv_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width) {
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const size_t target = static_cast<size_t>(positions[row]) * kv_width + column;
    key_cache[row][target] = k[index];
    value_cache[row][target] = v[index];
}

__global__ void store_kv_int8_batch_ptrs_kernel(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    const size_t scale_index =
        static_cast<size_t>(positions[row]) * kv_heads + head;
    const size_t cache_base = scale_index * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        key_scales[row][scale_index] = key_scale;
        value_scales[row][scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_cache[row][cache_base + d] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

void launch_store_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                     __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
                     int position, int kv_width, cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, position, nullptr, 0);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_device(const __nv_bfloat16* k, const __nv_bfloat16* v,
                            __nv_bfloat16* key_cache,
                            __nv_bfloat16* value_cache,
                            const int32_t* position, int kv_width,
                            cudaStream_t stream) {
    store_kv_kernel<<<(kv_width + 255) / 256, 256, 0, stream>>>(
        k, v, key_cache, value_cache, 1, kv_width, 0, position, 1);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_prefill(const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* key_cache,
                             __nv_bfloat16* value_cache,
                             int rows, int kv_width, cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, rows, kv_width, 0, nullptr, 2);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int8_t* key_cache, int8_t* value_cache,
                          float* key_scales, float* value_scales,
                          int position, int kv_heads, int head_dim,
                          cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, position, nullptr, 0);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_device(const __nv_bfloat16* k,
                                 const __nv_bfloat16* v,
                                 int8_t* key_cache, int8_t* value_cache,
                                 float* key_scales, float* value_scales,
                                 const int32_t* position, int kv_heads,
                                 int head_dim, cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, 1,
        kv_heads, head_dim, 0, position, 1);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_prefill(const __nv_bfloat16* k,
                                  const __nv_bfloat16* v,
                                  int8_t* key_cache, int8_t* value_cache,
                                  float* key_scales, float* value_scales,
                                  int rows, int kv_heads, int head_dim,
                                  cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales, rows,
        kv_heads, head_dim, 0, nullptr, 2);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

__global__ void store_kv_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    int kv_heads, int head_dim) {
    const size_t kv_width = static_cast<size_t>(kv_heads) * head_dim;
    const size_t total = static_cast<size_t>(rows) * kv_width;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int row = static_cast<int>(index / kv_width);
    const int column = static_cast<int>(index % kv_width);
    const int head = column / head_dim;
    const int dim = column % head_dim;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t target = paged_vector_offset(page, attention_slot, in_page,
                                               head, dim, page_tokens,
                                               page_vector_elements, layer_vector_offset, kv_heads,
                                               head_dim);
    key_pool[target] = k[index];
    value_pool[target] = v[index];
}

__global__ void store_kv_int8_paged_batch_kernel(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    size_t page_scale_elements, size_t layer_scale_offset,
    int kv_heads, int head_dim) {
    const int flat = blockIdx.x;
    const int row = flat / kv_heads;
    const int head = flat % kv_heads;
    if (row >= rows) return;
    const int position = positions[row];
    const int logical_page = position / page_tokens;
    const int in_page = position % page_tokens;
    const uint32_t page = page_tables[static_cast<size_t>(row) * page_table_stride + logical_page];
    const size_t source_base =
        (static_cast<size_t>(row) * kv_heads + head) * head_dim;
    float key_max = 0.0f;
    float value_max = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        key_max = fmaxf(key_max, fabsf(bf16_float(k[source_base + d])));
        value_max = fmaxf(value_max, fabsf(bf16_float(v[source_base + d])));
    }
    __shared__ float warp_values[32];
    __shared__ float key_total;
    __shared__ float value_total;
    key_max = block_max(key_max, warp_values, &key_total);
    value_max = block_max(value_max, warp_values, &value_total);
    __shared__ float key_scale;
    __shared__ float value_scale;
    if (threadIdx.x == 0) {
        key_scale = key_max > 0.0f ? key_max / 127.0f : 1.0f;
        value_scale = value_max > 0.0f ? value_max / 127.0f : 1.0f;
        const size_t scale_index = paged_scale_offset(
            page, attention_slot, in_page, head, page_tokens,
            page_scale_elements, layer_scale_offset, kv_heads);
        key_scale_pool[scale_index] = key_scale;
        value_scale_pool[scale_index] = value_scale;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const size_t target = paged_vector_offset(
            page, attention_slot, in_page, head, d, page_tokens,
            page_vector_elements, layer_vector_offset, kv_heads, head_dim);
        key_pool[target] = quantize_symmetric_int8(
            bf16_float(k[source_base + d]), key_scale);
        value_pool[target] = quantize_symmetric_int8(
            bf16_float(v[source_base + d]), value_scale);
    }
}

void launch_store_kv_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions,
    int rows,
    int kv_width,
    cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * kv_width;
    store_kv_batch_ptrs_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        k, v, key_cache, value_cache, positions, rows, kv_width);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_int8_batch_ptrs(
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int8_t* const* key_cache,
    int8_t* const* value_cache,
    float* const* key_scales,
    float* const* value_scales,
    const int32_t* positions,
    int rows,
    int kv_heads,
    int head_dim,
    cudaStream_t stream) {
    const int threads = attention_threads(head_dim);
    store_kv_int8_batch_ptrs_kernel<<<rows * kv_heads, threads, 0, stream>>>(
        k, v, key_cache, value_cache, key_scales, value_scales,
        positions, rows, kv_heads, head_dim);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_store_kv_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    int kv_heads, int head_dim,
    cudaStream_t stream) {
    const size_t total = static_cast<size_t>(rows) * kv_heads * head_dim;
    store_kv_paged_batch_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, stream>>>(
        k, v, key_pool, value_pool, page_tables, page_table_stride,
        positions, rows, attention_slot, page_tokens, page_vector_elements,
        layer_vector_offset,
        kv_heads, head_dim);
    CELEG_KERNEL_CHECK();
}

void launch_store_kv_int8_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, size_t page_vector_elements, size_t layer_vector_offset,
    size_t page_scale_elements, size_t layer_scale_offset,
    int kv_heads, int head_dim,
    cudaStream_t stream) {
    store_kv_int8_paged_batch_kernel<<<rows * kv_heads, 64, 0, stream>>>(
        k, v, key_pool, value_pool, key_scale_pool, value_scale_pool,
        page_tables, page_table_stride, positions, rows, attention_slot,
        page_tokens, page_vector_elements, layer_vector_offset,
        page_scale_elements, layer_scale_offset, kv_heads, head_dim);
    CELEG_KERNEL_CHECK();
}
