#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises rmsnorm and SwiGLU activation kernels.
void run_norm_activation_tests(celeg::CudaStream& stream);
}
