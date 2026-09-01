#include "kernels/kernels.cuh"
#include "support/assertions.hpp"
#include "support/cuda_kernel_assertions.cuh"
#include "utils.cuh"

#include <array>
#include <cmath>
#include <vector>

namespace {

constexpr int kRows = 3;
constexpr int kHeadDim = 2;

const std::array<float, 6> kExpected{
    1.0f, 10.0f,
    4.0f / 3.0f, 40.0f / 3.0f,
    11.0f / 7.0f, 110.0f / 7.0f};

void check_output(const celeg::DeviceBuffer<__nv_bfloat16>& output,
                  celeg::CudaStream& stream) {
    std::array<__nv_bfloat16, 6> host{};
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (size_t i = 0; i < host.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(
            celeg::cuda_test::to_float(host[i]) - kExpected[i]) < 0.04f);
    }
}

celeg::RelativePositionBiasDeviceView make_bias_view(
    const celeg::DeviceBuffer<float>& bias) {
    return {
        .values = bias.data(),
        .bucket_count = 4,
        .max_distance = 4,
        .bidirectional = false};
}

void run_bf16(celeg::CudaStream& stream,
              const celeg::DeviceBuffer<float>& bias) {
    const std::vector<__nv_bfloat16> query(6, celeg::cuda_test::to_bf16(0.0f));
    const std::vector<__nv_bfloat16> keys(6, celeg::cuda_test::to_bf16(0.0f));
    const std::vector<__nv_bfloat16> values{
        celeg::cuda_test::to_bf16(1.0f), celeg::cuda_test::to_bf16(10.0f),
        celeg::cuda_test::to_bf16(2.0f), celeg::cuda_test::to_bf16(20.0f),
        celeg::cuda_test::to_bf16(3.0f), celeg::cuda_test::to_bf16(30.0f)};

    celeg::DeviceBuffer<__nv_bfloat16> dquery(query.size());
    celeg::DeviceBuffer<__nv_bfloat16> dkeys(keys.size());
    celeg::DeviceBuffer<__nv_bfloat16> dvalues(values.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(values.size());
    CELEG_CUDA(cudaMemcpy(dquery.data(), query.data(), dquery.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkeys.data(), keys.data(), dkeys.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(),
                          cudaMemcpyHostToDevice));

    celeg::launch_gqa_prefill_relative({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(), .values = dvalues.data()},
        .out = output.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = kHeadDim},
        .extent = {.rows = kRows},
        .relative_bias = make_bias_view(bias),
        .stream = stream.get()});
    check_output(output, stream);
}

void run_int8(celeg::CudaStream& stream,
              const celeg::DeviceBuffer<float>& bias) {
    const std::vector<__nv_bfloat16> query(6, celeg::cuda_test::to_bf16(0.0f));
    const std::array<int8_t, 6> keys{};
    const std::array<int8_t, 6> values{1, 10, 2, 20, 3, 30};
    const std::array<float, 3> scales{1.0f, 1.0f, 1.0f};

    celeg::DeviceBuffer<__nv_bfloat16> dquery(query.size());
    celeg::DeviceBuffer<int8_t> dkeys(keys.size());
    celeg::DeviceBuffer<int8_t> dvalues(values.size());
    celeg::DeviceBuffer<float> key_scales(scales.size());
    celeg::DeviceBuffer<float> value_scales(scales.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(values.size());
    CELEG_CUDA(cudaMemcpy(dquery.data(), query.data(), dquery.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkeys.data(), keys.data(), dkeys.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(key_scales.data(), scales.data(), key_scales.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(value_scales.data(), scales.data(), value_scales.bytes(),
                          cudaMemcpyHostToDevice));

    celeg::launch_gqa_prefill_relative_int8({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(),
               .values = dvalues.data(),
               .key_scales = key_scales.data(),
               .value_scales = value_scales.data()},
        .out = output.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = kHeadDim},
        .extent = {.rows = kRows},
        .relative_bias = make_bias_view(bias),
        .stream = stream.get()});
    check_output(output, stream);
}

}

int main() {
    celeg::CudaStream stream;
    const std::array<float, 4> host_bias{
        0.0f, std::log(2.0f), std::log(4.0f), std::log(8.0f)};
    celeg::DeviceBuffer<float> bias(host_bias.size());
    CELEG_CUDA(cudaMemcpy(bias.data(), host_bias.data(), bias.bytes(),
                          cudaMemcpyHostToDevice));

    run_bf16(stream, bias);
    run_int8(stream, bias);
    return 0;
}
