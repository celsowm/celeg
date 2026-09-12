#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises the fused gated-delta-net prefill/decode kernels.
void run_gated_delta_tests(celeg::CudaStream& stream);
}
