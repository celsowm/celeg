
struct BatchPtrBf16Row {
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int kv_heads;

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const __nv_bfloat16* key = keys +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return attention_dot(query, key, head_dim, warp_sums, dot_total);
    }

    __device__ __forceinline__ float value(
        int token, int kv_head, int dimension, int head_dim) const {
        const __nv_bfloat16* value_ptr = values +
            (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
        return bf16_float(value_ptr[dimension]);
    }
};

struct BatchPtrBf16Storage {
    const __nv_bfloat16* const* keys;
    const __nv_bfloat16* const* values;
    int kv_heads;

    __device__ __forceinline__ BatchPtrBf16Row row(int index) const {
        return {keys[index], values[index], kv_heads};
    }
};

struct BatchPtrInt8Row {
    const int8_t* keys;
    const int8_t* values;
    const float* key_scales;
    const float* value_scales;
    int kv_heads;

    __device__ __forceinline__ float dot(
        const __nv_bfloat16* query, int token, int kv_head, int head_dim,
        float* warp_sums, float* dot_total) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* key = keys + scale_index * head_dim;
        return attention_dot_int8(
            query, key, key_scales[scale_index], head_dim, warp_sums, dot_total);
    }

    __device__ __forceinline__ float value(
        int token, int kv_head, int dimension, int head_dim) const {
        const size_t scale_index = static_cast<size_t>(token) * kv_heads + kv_head;
        const int8_t* value_ptr = values + scale_index * head_dim;
        return static_cast<float>(value_ptr[dimension]) * value_scales[scale_index];
    }
};

struct BatchPtrInt8Storage {
    const int8_t* const* keys;
    const int8_t* const* values;
    const float* const* key_scales;
    const float* const* value_scales;
    int kv_heads;

    __device__ __forceinline__ BatchPtrInt8Row row(int index) const {
        return {keys[index], values[index], key_scales[index],
                value_scales[index], kv_heads};
    }
};

template <typename Storage>
__global__ void gqa_decode_online_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    Storage storage,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const auto row_storage = storage.row(row);
    float running_max = -FLT_MAX;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float next_max;
    __shared__ float shared_denominator;
    for (int token = first_token; token < seq_len; ++token) {
        const float dot = row_storage.dot(
            query, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = dot * scale;
            next_max = fmaxf(running_max, score);
            alpha = expf(running_max - next_max);
            beta = expf(score - next_max);
            shared_denominator = denominator * alpha + beta;
        }
        __syncthreads();
        if (lane < head_dim) {
            accumulator = accumulator * alpha +
                row_storage.value(token, kv_head, lane, head_dim) * beta;
        }
        denominator = shared_denominator;
        running_max = next_max;
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator / denominator);
    }
}

template <typename Storage>
__global__ void gqa_decode_strict_batch_ptrs_kernel(
    const __nv_bfloat16* q,
    Storage storage,
    __nv_bfloat16* out,
    const int32_t* positions,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim, int sliding_window) {
    const int block = blockIdx.x;
    const int row = block / q_heads;
    const int query_head = block % q_heads;
    if (row >= rows) return;
    const int seq_len = positions[row] + 1;
    const int first_token = sliding_window > 0 ? max(0, seq_len - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query = q +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const auto row_storage = storage.row(row);
    const float scale = rsqrtf(static_cast<float>(head_dim));
    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;
    if (lane == 0) maximum = -FLT_MAX;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const float dot = row_storage.dot(
            query, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            maximum = fmaxf(maximum, score);
        }
        __syncthreads();
    }
    if (lane == 0) denominator = 0.0f;
    __syncthreads();
    for (int token = first_token; token < seq_len; ++token) {
        const float dot = row_storage.dot(
            query, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            denominator += expf(score - maximum);
        }
        __syncthreads();
    }
    float accumulator = 0.0f;
    for (int token = first_token; token < seq_len; ++token) {
        const float dot = row_storage.dot(
            query, token, kv_head, head_dim, warp_sums, &dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(rounded_bf16_float(dot) * scale);
            probability = rounded_bf16_float(expf(score - maximum) / denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            accumulator += probability *
                row_storage.value(token, kv_head, lane, head_dim);
        }
        __syncthreads();
    }
    if (lane < head_dim) {
        out[(static_cast<size_t>(row) * q_heads + query_head) * head_dim + lane] =
            __float2bfloat16(accumulator);
    }
}

void launch_gqa_decode_batch_ptrs(const GqaBatchPtrArgs& args) {
    const GqaGeometry& g = args.geometry;
    const int threads = attention_threads(g.head_dim);
    const BatchPtrBf16Storage storage{
        args.kv.keys, args.kv.values, g.kv_heads};
    if (args.fast) {
        gqa_decode_online_batch_ptrs_kernel<<<
            args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, storage, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_strict_batch_ptrs_kernel<<<
            args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, storage, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}

void launch_gqa_decode_int8_batch_ptrs(const GqaBatchPtrInt8Args& args) {
    const GqaGeometry& g = args.geometry;
    const int threads = attention_threads(g.head_dim);
    const BatchPtrInt8Storage storage{
        args.kv.keys, args.kv.values, args.kv.key_scales,
        args.kv.value_scales, g.kv_heads};
    if (args.fast) {
        gqa_decode_online_batch_ptrs_kernel<<<
            args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, storage, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    } else {
        gqa_decode_strict_batch_ptrs_kernel<<<
            args.rows * g.q_heads, threads, 0, args.stream>>>(
            args.query, storage, args.out, args.positions, args.rows,
            g.q_heads, g.kv_heads, g.head_dim, g.sliding_window);
    }
    CELEG_KERNEL_DEBUG_SYNC(args.stream);
}
