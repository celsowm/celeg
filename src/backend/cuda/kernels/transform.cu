#include "kernel_common.cuh"

// Pull in the shared GEMV kernel definitions (bf16_gemv_kernel,
// w8a16_gemv_kernel) before entering namespace celeg so the __global__ kernels
// live in the global namespace, not inside celeg.
#include "celeg/backend/cuda/kernels/gemv_kernels.cuh"
#include "gemv_launch.hpp"

namespace celeg {

void launch_bf16_gemv(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                      __nv_bfloat16* y, int n, int k, float beta,
                      cudaStream_t stream) {
    constexpr int warps_per_block = 8;
    const dim3 grid(static_cast<unsigned>(
        (n + warps_per_block - 1) / warps_per_block));
    bf16_gemv_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
        x, weight, y, n, k, beta);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

#include "linear.cuh"
#include "norm.cuh"
#include "shortconv_transform.cuh"
#include "rope.cuh"
}
