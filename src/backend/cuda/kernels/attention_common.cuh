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

__device__ __forceinline__ float merge_segmented_attention_lane(
    size_t base, int count, int lane, int head_dim,
    const float* partial_max, const float* partial_denom,
    const float* partial_accum) {
    float global_max = -FLT_MAX;
    for (int segment = 0; segment < count; ++segment) {
        global_max = fmaxf(global_max, partial_max[base + segment]);
    }
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (int segment = 0; segment < count; ++segment) {
        const float local_denom = partial_denom[base + segment];
        if (local_denom == 0.0f) continue;
        const float factor = expf(partial_max[base + segment] - global_max);
        denominator += local_denom * factor;
        if (lane < head_dim) {
            const size_t accum_index =
                (base + segment) * static_cast<size_t>(head_dim) + lane;
            accumulator += partial_accum[accum_index] * factor;
        }
    }
    return accumulator / denominator;
}
