#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises miscellaneous graph-capture-safe kernels.
void run_misc_tests(celeg::CudaStream& stream);
}
