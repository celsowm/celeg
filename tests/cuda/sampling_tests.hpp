#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
void run_sampling_tests(celeg::CudaStream& stream);
}
