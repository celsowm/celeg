#include "kernel_common.cuh"

// Pull in the shared GEMV kernel definitions (bf16_gemv_kernel,
// w8a16_gemv_kernel) before entering namespace lfm so the __global__ kernels
// live in the global namespace, not inside lfm.
#include "lfm/backend/cuda/kernels/gemv_kernels.cuh"

namespace lfm {
#include "transform.inl"
}
