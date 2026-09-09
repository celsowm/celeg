#pragma once

template <typename Storage>
__device__ __forceinline__ void block_sparse_attention_row(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* output,
    int query_position, int kv_head, int head_dim,
    const GqaBlockSparsePattern& pattern, float* warp_sums, float* dot_total,
    float* maximum, float* denominator, float* probability) {
    const int lane = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (lane == 0) *maximum = -FLT_MAX;
    __syncthreads();
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *maximum = fmaxf(*maximum, score);
        }
        __syncthreads();
    }

    if (lane == 0) *denominator = 0.0f;
    __syncthreads();
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *denominator += expf(score - *maximum);
        }
        __syncthreads();
    }

    float accumulator = 0.0f;
    for (int token = 0; token <= query_position; ++token) {
        if (!attention_block_sparse_visible(query_position, token, pattern)) continue;
        const float dot = storage.dot(
            query, token, kv_head, head_dim, warp_sums, dot_total);
        if (lane == 0) {
            const float score = rounded_bf16_float(
                rounded_bf16_float(dot) * scale);
            *probability = rounded_bf16_float(
                expf(score - *maximum) / *denominator);
        }
        __syncthreads();
        if (lane < head_dim) {
            accumulator += storage.block_sparse_weighted_value(
                *probability, token, kv_head, lane, head_dim);
        }
        __syncthreads();
    }

    if (lane < head_dim) {
        output[lane] = __float2bfloat16(accumulator);
    }
}

template <typename Storage, typename Positions>
__global__ void gqa_block_sparse_kernel(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* output,
    Positions positions, int rows, int q_heads, int kv_heads, int head_dim,
    GqaBlockSparsePattern pattern) {
    const int flat = blockIdx.x;
    const int row = flat / q_heads;
    const int query_head = flat % q_heads;
    if (row >= rows || query_head >= q_heads) return;

    const int query_position = positions.value(row);
    const int kv_head = query_head / (q_heads / kv_heads);
    const __nv_bfloat16* query_row = query +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    __nv_bfloat16* output_row = output +
        (static_cast<size_t>(row) * q_heads + query_head) * head_dim;
    const auto row_storage = storage.row(row);

    __shared__ float warp_sums[32];
    __shared__ float dot_total;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    block_sparse_attention_row(
        query_row, row_storage, output_row, query_position, kv_head, head_dim,
        pattern, warp_sums, &dot_total, &maximum, &denominator, &probability);
}

template <typename Storage, typename Positions>
inline void launch_block_sparse(
    const __nv_bfloat16* query, Storage storage, __nv_bfloat16* output,
    Positions positions, int rows, const GqaGeometry& geometry,
    GqaBlockSparsePattern pattern, cudaStream_t stream) {
    const int threads = attention_threads(geometry.head_dim);
    gqa_block_sparse_kernel<<<
        rows * geometry.q_heads, threads, 0, stream>>>(
        query, storage, output, positions, rows, geometry.q_heads,
        geometry.kv_heads, geometry.head_dim, pattern);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}
