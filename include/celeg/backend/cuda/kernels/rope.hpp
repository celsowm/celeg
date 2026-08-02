#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

// Generic per-layer RoPE/QK normalization. The table-free form is required
// when layers have different head dimensions, rotary fractions, or bases.
void launch_dynamic_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, int position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    cudaStream_t stream);
void launch_dynamic_qk_norm_rope_device(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int q_heads, int kv_heads, int head_dim, const int32_t* position,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    cudaStream_t stream);
void launch_dynamic_qk_norm_rope_prefill(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
    int rows, int q_heads, int kv_heads, int head_dim,
    float rope_theta, float rotary_fraction, float eps, bool normalize,
    cudaStream_t stream);

} // namespace celeg
