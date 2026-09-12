#include "misc_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"

#include <cstdint>

namespace celeg::cuda_test {

void run_misc_tests(celeg::CudaStream& stream) {
{
    int32_t initial = 7;
    celeg::DeviceBuffer<int32_t> position(1);
    CELEG_CUDA(cudaMemcpy(position.data(), &initial, sizeof(initial), cudaMemcpyHostToDevice));
    celeg::CudaGraphExec graph;
    graph.capture_begin(stream.get());
    celeg::launch_increment_position(position.data(), stream.get());
    celeg::launch_increment_position(position.data(), stream.get());
    graph.capture_end(stream.get());
    graph.launch(stream.get());
    graph.launch(stream.get());
    int32_t result = 0;
    CELEG_CUDA(cudaMemcpyAsync(&result, position.data(), sizeof(result),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(result == 11);
}
}

}
