#include "celeg/attention/bias_semantics.hpp"
#include "support/assertions.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace {

struct BiasCase {
    int query_position;
    int key_position;
    int bucket_count;
    int max_distance;
    bool bidirectional;
    float slope;
};

__global__ void bias_semantics_probe(const BiasCase* cases, int* buckets,
                                     float* alibi, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    const BiasCase c = cases[index];
    buckets[index] = celeg::attention_semantics::relative_position_bucket(
        c.query_position, c.key_position, c.bucket_count,
        c.max_distance, c.bidirectional);
    alibi[index] = celeg::attention_semantics::alibi_bias(
        c.slope, c.query_position, c.key_position);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}

int main() {
    constexpr std::array<BiasCase, 8> cases{{
        {0, 0, 32, 128, false, 0.5f},
        {7, 0, 32, 128, false, 0.5f},
        {31, 15, 32, 128, false, 0.25f},
        {127, 0, 32, 128, false, 1.0f},
        {8, 9, 32, 128, true, 0.5f},
        {8, 7, 32, 128, true, 0.5f},
        {64, 127, 32, 128, true, 0.125f},
        {127, 64, 32, 128, true, 0.125f},
    }};

    BiasCase* device_cases = nullptr;
    int* device_buckets = nullptr;
    float* device_alibi = nullptr;
    check_cuda(cudaMalloc(&device_cases, sizeof(cases)));
    check_cuda(cudaMalloc(&device_buckets, cases.size() * sizeof(int)));
    check_cuda(cudaMalloc(&device_alibi, cases.size() * sizeof(float)));
    check_cuda(cudaMemcpy(device_cases, cases.data(), sizeof(cases),
                          cudaMemcpyHostToDevice));

    bias_semantics_probe<<<1, 32>>>(device_cases, device_buckets,
                                   device_alibi,
                                   static_cast<int>(cases.size()));
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());

    std::array<int, cases.size()> buckets{};
    std::array<float, cases.size()> alibi{};
    check_cuda(cudaMemcpy(buckets.data(), device_buckets,
                          buckets.size() * sizeof(int), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(alibi.data(), device_alibi,
                          alibi.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t index = 0; index < cases.size(); ++index) {
        const BiasCase& c = cases[index];
        CELEG_TEST_CHECK(buckets[index] ==
            celeg::attention_semantics::relative_position_bucket(
                c.query_position, c.key_position, c.bucket_count,
                c.max_distance, c.bidirectional));
        CELEG_TEST_CHECK(std::abs(alibi[index] -
            celeg::attention_semantics::alibi_bias(
                c.slope, c.query_position, c.key_position)) < 1.0e-6f);
    }

    CELEG_TEST_CHECK(alibi[4] == alibi[5]);
    CELEG_TEST_CHECK(buckets[4] != buckets[5]);

    cudaFree(device_alibi);
    cudaFree(device_buckets);
    cudaFree(device_cases);
    return 0;
}
