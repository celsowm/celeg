#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace celeg {

void launch_mamba2_step(const __nv_bfloat16* projected,
                        const __nv_bfloat16* conv_weight,
                        const __nv_bfloat16* conv_bias,
                        const __nv_bfloat16* dt_bias,
                        const __nv_bfloat16* a_log,
                        const __nv_bfloat16* d,
                        __nv_bfloat16* conv_state,
                        __nv_bfloat16* ssm_state,
                        __nv_bfloat16* inner,
                        int intermediate, int state_size, int num_heads,
                        int head_dim, int group_count, int conv_kernel,
                        cudaStream_t stream);

} // namespace celeg
