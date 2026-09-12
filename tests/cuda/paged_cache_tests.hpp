#pragma once

namespace celeg {
class CudaStream;
}

namespace celeg::cuda_test {
/// Exercises PhysicalPagedKvCache clone/prefix-clone paths.
void run_paged_cache_tests(celeg::CudaStream& stream);
}
