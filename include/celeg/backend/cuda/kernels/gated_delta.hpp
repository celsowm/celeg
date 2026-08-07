#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace celeg {

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
                            int value_heads, float eps, cudaStream_t stream);

} // namespace celeg
