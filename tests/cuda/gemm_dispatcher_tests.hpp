#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises GemmDispatcher FP8/NVFP4 end-to-end paths.
void run_gemm_dispatcher_tests(celeg::CudaStream& stream);
}
