#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace celeg {

void launch_bf16_gemv(const __nv_bfloat16* x,
                      const __nv_bfloat16* weight,
                      __nv_bfloat16* y,
                      int n,
                      int k,
                      float beta,
                      cudaStream_t stream);

}
