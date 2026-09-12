#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises W4A16/W8A16 quantized linear kernels.
void run_quantized_linear_tests(celeg::CudaStream& stream);
}
