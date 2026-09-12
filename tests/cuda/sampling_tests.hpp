#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
void run_sampling_tests(celeg::CudaStream& stream);
/// Exercises packed batched sampling and seen-mask helpers.
void run_packed_sampling_tests(celeg::CudaStream& stream);
}
