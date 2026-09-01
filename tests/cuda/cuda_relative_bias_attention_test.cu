#include "celeg/backend/cpu/paged_kv.hpp"
#include "kernels/kernels.cuh"
#include "support/assertions.hpp"
#include "support/cuda_kernel_assertions.cuh"
#include "utils.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

constexpr int kRows = 3;
constexpr int kHeadDim = 2;
constexpr int kBucketCount = 4;
constexpr int kMaxDistance = 4;

const std::array<float, 6> kValues{
    1.0f, 10.0f,
    2.0f, 20.0f,
    3.0f, 30.0f};

std::array<float, 6> cpu_reference(
    const std::array<float, kBucketCount>& bias_values) {
    const celeg::RelativePositionBiasSpec spec{
        kBucketCount, kMaxDistance, false};
    const celeg::CpuAttentionBias bias = celeg::CpuAttentionBias::lower(
        spec, bias_values, 1);

    std::array<float, 6> output{};
    for (int row = 0; row < kRows; ++row) {
        float maximum = -INFINITY;
        for (int token = 0; token <= row; ++token) {
            maximum = std::max(maximum, bias.score(0, row, token));
        }
        float denominator = 0.0f;
        std::array<float, kHeadDim> accumulator{};
        for (int token = 0; token <= row; ++token) {
            const float weight = std::exp(
                bias.score(0, row, token) - maximum);
            denominator += weight;
            for (int d = 0; d < kHeadDim; ++d) {
                accumulator[static_cast<size_t>(d)] +=
                    kValues[static_cast<size_t>(token * kHeadDim + d)] * weight;
            }
        }
        for (int d = 0; d < kHeadDim; ++d) {
            output[static_cast<size_t>(row * kHeadDim + d)] =
                accumulator[static_cast<size_t>(d)] / denominator;
        }
    }
    return output;
}

void check_output(const celeg::DeviceBuffer<__nv_bfloat16>& output,
                  celeg::CudaStream& stream,
                  const std::array<float, 6>& expected) {
    std::array<__nv_bfloat16, 6> host{};
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (size_t i = 0; i < host.size(); ++i) {
        CELEG_TEST_CHECK(std::abs(
            celeg::cuda_test::to_float(host[i]) - expected[i]) < 0.04f);
    }
}

celeg::RelativePositionBiasDeviceView make_bias_view(
    const celeg::DeviceBuffer<float>& bias) {
    return {
        .values = bias.data(),
        .bucket_count = kBucketCount,
        .max_distance = kMaxDistance,
        .bidirectional = false};
}

void run_bf16(celeg::CudaStream& stream,
              const celeg::DeviceBuffer<float>& bias,
              const std::array<float, 6>& expected) {
    const std::vector<__nv_bfloat16> query(6, celeg::cuda_test::to_bf16(0.0f));
    const std::vector<__nv_bfloat16> keys(6, celeg::cuda_test::to_bf16(0.0f));
    std::vector<__nv_bfloat16> values;
    values.reserve(kValues.size());
    for (float value : kValues) values.push_back(celeg::cuda_test::to_bf16(value));

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
    check_output(output, stream, expected);
}

void run_int8(celeg::CudaStream& stream,
              const celeg::DeviceBuffer<float>& bias,
              const std::array<float, 6>& expected) {
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
    check_output(output, stream, expected);
}

}

int main() {
    celeg::CudaStream stream;
    const std::array<float, kBucketCount> host_bias{
        0.0f, std::log(2.0f), std::log(4.0f), std::log(8.0f)};
    const std::array<float, 6> expected = cpu_reference(host_bias);
    celeg::DeviceBuffer<float> bias(host_bias.size());
    CELEG_CUDA(cudaMemcpy(bias.data(), host_bias.data(), bias.bytes(),
                          cudaMemcpyHostToDevice));

    run_bf16(stream, bias, expected);
    run_int8(stream, bias, expected);
    return 0;
}
