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
    CudaRopeScaling result{};
    std::visit([&](const auto& scaling) {
        using Scaling = std::decay_t<decltype(scaling)>;
        if constexpr (std::is_same_v<Scaling, NoRopeScaling>) {
            result.kind = 0;
        } else if constexpr (std::is_same_v<Scaling, LinearRopeScaling>) {
            result.kind = 1;
            result.factor = static_cast<float>(scaling.factor);
        } else if constexpr (std::is_same_v<Scaling, DynamicNtkRopeScaling>) {
            result.kind = 2;
            result.factor = static_cast<float>(scaling.factor);
            result.original_context = scaling.original_context;
        } else if constexpr (std::is_same_v<Scaling, YarnRopeScaling>) {
            result.kind = 3;
            result.factor = static_cast<float>(scaling.factor);
            result.original_context = scaling.original_context;
            result.attention_factor = static_cast<float>(scaling.attention_factor);
            result.beta_fast = static_cast<float>(scaling.beta_fast);
            result.beta_slow = static_cast<float>(scaling.beta_slow);
        } else if constexpr (std::is_same_v<Scaling, LongRopeScaling>) {
            result.kind = 4;
            result.original_context = scaling.original_context;
            if (scaling.short_factors.size() != scaling.long_factors.size() ||
                scaling.short_factors.size() > 128) {
                throw std::invalid_argument("LongRoPE factors exceed CUDA lowering capacity");
            }
            result.factor_count = static_cast<int>(scaling.short_factors.size());
            for (int i = 0; i < result.factor_count; ++i) {
                result.short_factors[i] = static_cast<float>(scaling.short_factors[i]);
                result.long_factors[i] = static_cast<float>(scaling.long_factors[i]);
            }
        } else if constexpr (std::is_same_v<Scaling, Llama3FrequencyScaling>) {
            result.kind = 5;
            result.factor = static_cast<float>(scaling.factor);
            result.original_context = scaling.original_context;
            result.low_frequency_factor = static_cast<float>(scaling.low_frequency_factor);
            result.high_frequency_factor = static_cast<float>(scaling.high_frequency_factor);
        } else {
            static_assert(always_false_v<Scaling>, "unhandled RoPE scaling variant");
        }
    }, rope.scaling);
    return result;
}

void launch_dynamic_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, int position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling, RopePairingKind pairing,
    cudaStream_t stream);
void launch_dynamic_qk_norm_rope_device(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, const int32_t* position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    CudaRopeScaling scaling, RopePairingKind pairing,
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
    CudaRopeScaling scaling, RopePairingKind pairing,
    cudaStream_t stream);

}
