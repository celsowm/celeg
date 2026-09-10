#include "celeg/attention/pattern_semantics.hpp"
#include "support/assertions.hpp"
#include "support/cuda_kernel_assertions.cuh"

#include <array>

namespace {

struct Result {
    int causal;
    int sliding;
    int block_sparse;
    int prefix;
    int first_candidate;
};

__global__ void probe(Result* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    out->causal = celeg::attention_semantics::causal_visible(8, 7);
    out->sliding = celeg::attention_semantics::sliding_window_visible(8, 5, 4);
    out->block_sparse = celeg::attention_semantics::block_sparse_visible(
        63, 32, 16, 2, 1);
    out->prefix = celeg::attention_semantics::prefix_lm_visible(2, 5, 6);
    out->first_candidate =
        celeg::attention_semantics::sliding_window_first_candidate(8, 4);
}

}

int main() {
    Result* device = nullptr;
    CELEG_CUDA(cudaMalloc(&device, sizeof(Result)));
    probe<<<1, 1>>>(device);
    CELEG_CUDA(cudaGetLastError());
    Result host{};
    CELEG_CUDA(cudaMemcpy(&host, device, sizeof(Result), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaFree(device));

    CELEG_TEST_CHECK(host.causal == 1);
    CELEG_TEST_CHECK(host.sliding == 1);
    CELEG_TEST_CHECK(host.block_sparse == 1);
    CELEG_TEST_CHECK(host.prefix == 1);
    CELEG_TEST_CHECK(host.first_candidate == 5);
    return 0;
}
