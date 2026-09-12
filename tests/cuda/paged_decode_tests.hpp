#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises paged GQA decode kernels and KV store paths.
void run_paged_decode_tests(celeg::CudaStream& stream);
}
