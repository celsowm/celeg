#pragma once

#include "celeg/backend/cuda/kernels/norm_conv.hpp"
#include "celeg/model/graph.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace celeg {

/// Pre-scale factor to apply to Q before the regular (non-latent) attention
/// kernels. Those kernels bake `1/sqrt(head_dim)` into their score computation,
/// while AttentionSpec::query_scale is the absolute pre-softmax scale, so the
/// caller has to divide the kernel-side factor back out. Checkpoints that pin
/// an explicit attention_multiplier (Granite uses 1/head_dim rather than
/// 1/sqrt(head_dim)) otherwise end up scaled by their multiplier *and* by the
/// kernel default. The CPU backend performs the same division.
inline float cuda_query_prescale(const AttentionSpec& layout) {
    if (layout.head_dim <= 0) return layout.query_scale;
    return layout.query_scale * std::sqrt(static_cast<float>(layout.head_dim));
}

inline void launch_attention_norm(
    __nv_bfloat16* data,
    const __nv_bfloat16* weight,
    int rows,
    int heads,
    int head_dim,
    const NormSpec& norm,
    cudaStream_t stream) {
    if (data == nullptr || weight == nullptr) return;
    if (norm.granularity == NormGranularity::PerHead) {
        launch_rmsnorm(
            data, weight, data,
            rows * heads, head_dim, norm.epsilon, stream);
        return;
    }
    launch_rmsnorm(
        data, weight, data,
        rows, heads * head_dim, norm.epsilon, stream);
}

inline void launch_attention_qk_norm(
    const AttentionSpec& layout,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    const __nv_bfloat16* query_weight,
    const __nv_bfloat16* key_weight,
    int rows,
    cudaStream_t stream) {
    if (layout.query_norm) {
        launch_attention_norm(
            query, query_weight, rows, layout.query_heads,
            layout.head_dim, *layout.query_norm, stream);
    }
    if (layout.key_norm && key != nullptr) {
        launch_attention_norm(
            key, key_weight, rows, layout.key_value_heads,
            layout.head_dim, *layout.key_norm, stream);
    }
}

}