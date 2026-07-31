#include "kernel_common.cuh"

// Pull in the shared GEMV kernel definitions (bf16_gemv_kernel,
// w8a16_gemv_kernel) before entering namespace celeg so the __global__ kernels
// live in the global namespace, not inside celeg.
#include "celeg/backend/cuda/kernels/gemv_kernels.cuh"

namespace celeg {
#include "linear.cuh"
#include "norm.cuh"
#include "shortconv_transform.cuh"
#include "rope.cuh"
}
