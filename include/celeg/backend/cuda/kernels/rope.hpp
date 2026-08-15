#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "celeg/model/position.hpp"

namespace celeg {

struct CudaRopeScaling {
    int kind = 0;
    float factor = 1.0f;
    int original_context = 0;
    float attention_factor = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    float low_frequency_factor = 1.0f;
    float high_frequency_factor = 1.0f;
    int factor_count = 0;
    float short_factors[128]{};
    float long_factors[128]{};
};

inline CudaRopeScaling lower_cuda_rope_scaling(const RopePositionSpec& rope) {
    CudaRopeScaling result;
    result.kind = static_cast<int>(rope.scaling.kind);
    result.factor = static_cast<float>(rope.scaling.factor);
    result.original_context = rope.scaling.original_context;
    result.attention_factor = static_cast<float>(rope.scaling.attention_factor);
    result.beta_fast = static_cast<float>(rope.scaling.beta_fast);
    result.beta_slow = static_cast<float>(rope.scaling.beta_slow);
    result.low_frequency_factor = static_cast<float>(rope.scaling.low_frequency_factor);
    result.high_frequency_factor = static_cast<float>(rope.scaling.high_frequency_factor);
    if (rope.scaling.kind == RopeScalingKind::Long) {
        if (rope.scaling.short_factors.size() != rope.scaling.long_factors.size() ||
            rope.scaling.short_factors.size() > 128) {
            throw std::invalid_argument("LongRoPE factors exceed CUDA lowering capacity");
        }
        result.factor_count = static_cast<int>(rope.scaling.short_factors.size());
        for (int i = 0; i < result.factor_count; ++i) {
            result.short_factors[i] = static_cast<float>(rope.scaling.short_factors[i]);
            result.long_factors[i] = static_cast<float>(rope.scaling.long_factors[i]);
        }
    }
    return result;
}

void launch_dynamic_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, int position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling,
    cudaStream_t stream);
void launch_dynamic_qk_norm_rope_device(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, const int32_t* position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling,
    cudaStream_t stream);
void launch_dynamic_mrope_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim,
    const int32_t* positions, int section0, int section1, int section2,
    bool interleaved, float rope_theta, float rotary_fraction, float eps,
    bool normalize, CudaRopeScaling scaling, cudaStream_t stream);
void launch_dynamic_qk_norm_rope_prefill(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int rows, int q_heads, int kv_heads, int head_dim,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling,
    cudaStream_t stream);

}
