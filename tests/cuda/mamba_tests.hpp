#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises the Mamba-2 prefill/step kernels.
void run_mamba_tests(celeg::CudaStream& stream);
}
