#pragma once

__device__ __forceinline__ float attention_dot_int8(const __nv_bfloat16* query,
                                    const int8_t* key, float key_scale,
                                    int head_dim, float* warp_sums,
                                    float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) *
            (static_cast<float>(key[d]) * key_scale);
    }
    return block_sum(partial, warp_sums, total);
}

__device__ __forceinline__ float attention_dot(const __nv_bfloat16* query,
                               const __nv_bfloat16* key,
                               int head_dim,
                               float* warp_sums,
                               float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) * bf16_float(key[d]);
    }
    return block_sum(partial, warp_sums, total);
}

/// Online/segmented decode kernels launch with 32 threads and stride
/// `for (d = lane; d < head_dim; d += 32)`, accumulating into a per-lane
/// register array. The supported head_dim ceiling is therefore
/// `kMaxHeadDimPerLane * 32`; 16 covers head_dim 512 (e.g. Gemma full-attention
/// layers) and anything larger must fail in `validate_cuda_attention_semantics`.
constexpr int kMaxHeadDimPerLane = 16;

__device__ __forceinline__ float warp_broadcast_sum(float partial) {
    return __shfl_sync(0xffffffffu, warp_sum(partial), 0);
}
