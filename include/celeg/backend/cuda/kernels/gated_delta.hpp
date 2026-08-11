#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace celeg {

void launch_interleave_gated_delta_qkv(const __nv_bfloat16* q,
                                       const __nv_bfloat16* k,
                                       const __nv_bfloat16* v,
                                       __nv_bfloat16* qkv, int rows,
                                       int key_width, int value_width,
                                       cudaStream_t stream);

void launch_gated_delta_net(const __nv_bfloat16* projected_qkv,
                            const __nv_bfloat16* projected_z,
                            const __nv_bfloat16* projected_b,
                            const __nv_bfloat16* projected_a,
                            const __nv_bfloat16* conv_weight,
                            const __nv_bfloat16* dt_bias,
                            const __nv_bfloat16* a_log,
                            const __nv_bfloat16* norm_weight,
                            __nv_bfloat16* conv_state,
                            __nv_bfloat16* recurrent_state,
                            __nv_bfloat16* output, int rows, int conv_kernel,
                            int key_head_dim, int value_head_dim, int key_heads,
                            int value_heads, float eps, bool vector_decay,
                            bool safe_decay, float decay_lower_bound,
                            bool sigmoid_output_gate, cudaStream_t stream);

} // namespace celeg
