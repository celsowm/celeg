#include "quantized_linear_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"

#include <cstdint>
#include <vector>

namespace celeg::cuda_test {

void run_quantized_linear_tests(celeg::CudaStream& stream) {
{
    std::vector<__nv_bfloat16> x = {
        to_bf16(1.0f), to_bf16(1.0f),
        to_bf16(1.0f), to_bf16(1.0f)};
    std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
    std::vector<float> scales = {0.5f, 0.25f};
    celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(2);
    celeg::DeviceBuffer<uint8_t> dw(packed.size());
    celeg::DeviceBuffer<float> ds(scales.size());
    CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), packed.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_w4a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                             1, 2, 4, 0.0f, stream.get());
    std::vector<__nv_bfloat16> output(2);
    CELEG_CUDA(cudaMemcpyAsync(output.data(), dy.data(), dy.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    expect_near(to_float(output[0]), 5.0f, 0.02f);
    expect_near(to_float(output[1]), -2.5f, 0.02f);
}

{
    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;
    std::vector<__nv_bfloat16> x = {
        to_bf16(1), to_bf16(2), to_bf16(3), to_bf16(4),
        to_bf16(2), to_bf16(0), to_bf16(-1), to_bf16(1)};
    std::vector<int8_t> weight = {
        1, 2, 3, 4,
        -1, 1, -1, 1,
        1, 0, 0, 0};
    std::vector<float> scales = {0.5f, 2.0f, 3.0f};
    celeg::DeviceBuffer<__nv_bfloat16> dx(x.size()), dy(m * n);
    celeg::DeviceBuffer<int8_t> dw(weight.size());
    celeg::DeviceBuffer<float> ds(scales.size());
    CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_w8a16_linear(dx.data(), dw.data(), ds.data(), dy.data(),
                             m, n, k, 0.0f, stream.get());
    std::vector<__nv_bfloat16> result(m * n);
    CELEG_CUDA(cudaMemcpyAsync(result.data(), dy.data(), dy.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const std::vector<float> expected = {15.0f, 4.0f, 3.0f, 1.5f, 0.0f, 6.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        expect_near(to_float(result[i]), expected[i], 0.05f);
    }
}
}

}
