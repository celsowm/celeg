#include "celeg/attention/micro_semantics.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

__global__ void attention_micro_probe(int* integers, float* scale) {
    integers[0] = celeg::attention_semantics::gqa_kv_head(7, 8, 2);
    integers[1] = celeg::attention_semantics::sequence_length_from_query_position(31);
    integers[2] = celeg::attention_semantics::query_position_from_sequence_length(32);
    *scale = celeg::attention_semantics::attention_scale(64);
}

int main() {
    try {
        int* integers = nullptr;
        float* scale = nullptr;
        cudaMallocManaged(&integers, 3 * sizeof(int));
        cudaMallocManaged(&scale, sizeof(float));
        attention_micro_probe<<<1, 1>>>(integers, scale);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("CUDA micro semantics probe failed");
        }
        if (integers[0] != 1 || integers[1] != 32 || integers[2] != 31) {
            throw std::runtime_error("CUDA attention micro semantics drifted");
        }
        if (std::abs(*scale - 0.125f) > 1.0e-7f) {
            throw std::runtime_error("CUDA attention scale semantics drifted");
        }
        cudaFree(integers);
        cudaFree(scale);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
