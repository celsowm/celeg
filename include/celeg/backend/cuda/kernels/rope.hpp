#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_qk_norm_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                               const __nv_bfloat16* q_norm,
                               const __nv_bfloat16* k_norm,
                               const __nv_bfloat16* rope_cos,
                               const __nv_bfloat16* rope_sin,
                               int q_heads, int kv_heads, int head_dim,
                               int position, float eps,
                               cudaStream_t stream);
void launch_qk_norm_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                      const __nv_bfloat16* q_norm,
                                      const __nv_bfloat16* k_norm,
                                      const __nv_bfloat16* rope_cos,
                                      const __nv_bfloat16* rope_sin,
                                      int q_heads, int kv_heads, int head_dim,
                                      const int32_t* position, float eps,
                                      cudaStream_t stream);
void launch_qk_norm_rope_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                             const __nv_bfloat16* q_norm,
                             const __nv_bfloat16* k_norm,
                             const __nv_bfloat16* rope_cos,
                             const __nv_bfloat16* rope_sin,
                             int q_heads, int kv_heads, int head_dim,
                             int position, float eps,
                             cudaStream_t stream);
void launch_qk_norm_rope_fast_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                    const __nv_bfloat16* q_norm,
                                    const __nv_bfloat16* k_norm,
                                    const __nv_bfloat16* rope_cos,
                                    const __nv_bfloat16* rope_sin,
                                    int q_heads, int kv_heads, int head_dim,
                                    const int32_t* position, float eps,
                                    cudaStream_t stream);
void launch_qk_norm_rope_prefill_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                       const __nv_bfloat16* q_norm,
                                       const __nv_bfloat16* k_norm,
                                       const __nv_bfloat16* rope_cos,
                                       const __nv_bfloat16* rope_sin,
                                       int rows, int q_heads, int kv_heads,
                                       int head_dim, float eps,
                                       cudaStream_t stream);
void launch_qk_norm_rope_prefill_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                                     const __nv_bfloat16* q_norm,
                                     const __nv_bfloat16* k_norm,
                                     const __nv_bfloat16* rope_cos,
                                     const __nv_bfloat16* rope_sin,
                                     int rows, int q_heads, int kv_heads,
                                     int head_dim, float eps,
                                     cudaStream_t stream);
void launch_qk_norm_rope_batch_positions(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm,
    const __nv_bfloat16* k_norm,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions, int rows,
    int q_heads, int kv_heads, int head_dim,
    float eps, bool fast, cudaStream_t stream);
void launch_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                        const __nv_bfloat16* rope_cos,
                        const __nv_bfloat16* rope_sin,
                        int q_heads, int kv_heads, int head_dim,
                        int position, cudaStream_t stream);
void launch_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                               const __nv_bfloat16* rope_cos,
                               const __nv_bfloat16* rope_sin,
                               int q_heads, int kv_heads, int head_dim,
                               const int32_t* position, cudaStream_t stream);
void launch_rope_prefill(__nv_bfloat16* q, __nv_bfloat16* k,
                         const __nv_bfloat16* rope_cos,
                         const __nv_bfloat16* rope_sin,
                         int rows, int q_heads, int kv_heads, int head_dim,
                         cudaStream_t stream);
void launch_rope_batch_positions(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* rope_cos, const __nv_bfloat16* rope_sin,
    const int32_t* positions, int rows, int q_heads, int kv_heads,
    int head_dim, cudaStream_t stream);

} // namespace celeg
