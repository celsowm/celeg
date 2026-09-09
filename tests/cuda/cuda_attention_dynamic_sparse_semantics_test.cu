#include "celeg/attention/dynamic_sparse_semantics.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace {

__global__ void dynamic_sparse_probe(const float* candidate_scores,
                                     int candidate_count,
                                     int max_selected_blocks,
                                     int* blocks,
                                     float* selected_scores) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    celeg::attention_semantics::dynamic_sparse_select_top_k(
        candidate_scores, candidate_count, max_selected_blocks,
        blocks, selected_scores);
}

void run_case(const float* host_scores, int candidate_count, int k) {
    float* scores = nullptr;
    int* device_blocks = nullptr;
    float* device_selected_scores = nullptr;
    cudaMallocManaged(&scores, candidate_count * sizeof(float));
    cudaMallocManaged(&device_blocks, k * sizeof(int));
    cudaMallocManaged(&device_selected_scores, k * sizeof(float));
    for (int i = 0; i < candidate_count; ++i) scores[i] = host_scores[i];

    dynamic_sparse_probe<<<1, 1>>>(
        scores, candidate_count, k, device_blocks, device_selected_scores);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("CUDA dynamic sparse semantics probe failed");
    }

    int expected_blocks[8];
    float expected_scores[8];
    if (k > 8) throw std::runtime_error("dynamic sparse test capacity exceeded");
    celeg::attention_semantics::dynamic_sparse_select_top_k(
        host_scores, candidate_count, k, expected_blocks, expected_scores);
    for (int i = 0; i < k; ++i) {
        if (device_blocks[i] != expected_blocks[i] ||
            device_selected_scores[i] != expected_scores[i]) {
            throw std::runtime_error("CUDA dynamic sparse selection semantics drifted");
        }
    }

    cudaFree(scores);
    cudaFree(device_blocks);
    cudaFree(device_selected_scores);
}

}

int main() {
    try {
        const float ranked[] = {1.0f, 4.0f, 2.0f};
        run_case(ranked, 3, 2);
        const float tied[] = {3.0f, 3.0f, 2.0f};
        run_case(tied, 3, 2);
        const float negative[] = {-4.0f, -2.0f, -3.0f};
        run_case(negative, 3, 3);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
