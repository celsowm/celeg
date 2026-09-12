#include "norm_activation_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"

#include <cmath>
#include <vector>

namespace celeg::cuda_test {

void run_norm_activation_tests(celeg::CudaStream& stream) {
{
    std::vector<__nv_bfloat16> x = {
        to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f)};
    std::vector<__nv_bfloat16> weight(4, to_bf16(1.0f));
    celeg::DeviceBuffer<__nv_bfloat16> dx(4), dw(4), dy(4);
    CELEG_CUDA(cudaMemcpy(dx.data(), x.data(), dx.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dw.data(), weight.data(), dw.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_rmsnorm(dx.data(), dw.data(), dy.data(), 1, 4, 1e-5f, stream.get());
    std::vector<__nv_bfloat16> y(4);
    CELEG_CUDA(cudaMemcpyAsync(y.data(), dy.data(), dy.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const float inv = 1.0f / std::sqrt(7.5f + 1e-5f);
    for (int i = 0; i < 4; ++i) expect_near(to_float(y[i]), (i + 1) * inv);
}

{
    std::vector<__nv_bfloat16> gate_up = {
        to_bf16(0.0f), to_bf16(1.0f),
        to_bf16(2.0f), to_bf16(3.0f)};
    celeg::DeviceBuffer<__nv_bfloat16> input(4), output(2);
    CELEG_CUDA(cudaMemcpy(input.data(), gate_up.data(), input.bytes(),
                        cudaMemcpyHostToDevice));
    celeg::launch_swiglu_fused(input.data(), output.data(), 2, stream.get());
    std::vector<__nv_bfloat16> result(2);
    CELEG_CUDA(cudaMemcpyAsync(result.data(), output.data(), output.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    expect_near(to_float(result[0]), 0.0f);
    expect_near(to_float(result[1]), (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f);
}
}

}
