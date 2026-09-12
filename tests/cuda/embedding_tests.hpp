#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises embedding gather kernels across weight modes.
void run_embedding_tests(celeg::CudaStream& stream);
}
