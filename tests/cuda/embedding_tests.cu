#include "embedding_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"
#include "weight_layout.hpp"

#include <cstdint>
#include <vector>

namespace celeg::cuda_test {

void run_embedding_tests(celeg::CudaStream& stream) {
{
    std::vector<__nv_bfloat16> table(12);
    for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
    celeg::DeviceBuffer<__nv_bfloat16> device_table(table.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(4);
    CELEG_CUDA(cudaMemcpy(device_table.data(), table.data(),
                        table.size() * sizeof(table[0]),
                        cudaMemcpyHostToDevice));
    celeg::launch_embedding(2, device_table.data(), output.data(), 4, stream.get());
    std::vector<__nv_bfloat16> host(4);
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < 4; ++i) expect_near(to_float(host[i]), 8.0f + i);
}

{
    std::vector<__nv_bfloat16> table(12);
    for (int i = 0; i < 12; ++i) table[i] = to_bf16(static_cast<float>(i));
    std::vector<int32_t> tokens = {2, 0};
    celeg::DeviceBuffer<__nv_bfloat16> device_table(table.size());
    celeg::DeviceBuffer<int32_t> device_tokens(tokens.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(8);
    CELEG_CUDA(cudaMemcpy(device_table.data(), table.data(),
                        table.size() * sizeof(table[0]), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(device_tokens.data(), tokens.data(),
                        tokens.size() * sizeof(tokens[0]), cudaMemcpyHostToDevice));
    celeg::launch_embedding_batch(device_tokens.data(), 2, device_table.data(),
                                output.data(), 4, stream.get());
    std::vector<__nv_bfloat16> host(8);
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < 4; ++i) {
        expect_near(to_float(host[i]), 8.0f + i);
        expect_near(to_float(host[4 + i]), static_cast<float>(i));
    }
}

{
    constexpr int hidden = 4;
    std::vector<__nv_bfloat16> bf16_table = {
        to_bf16(1.0f), to_bf16(2.0f), to_bf16(3.0f), to_bf16(4.0f),
        to_bf16(5.0f), to_bf16(6.0f), to_bf16(7.0f), to_bf16(8.0f)};
    celeg::DeviceBuffer<__nv_bfloat16> dbf16(bf16_table.size()), out(hidden);
    CELEG_CUDA(cudaMemcpy(dbf16.data(), bf16_table.data(), dbf16.bytes(),
                          cudaMemcpyHostToDevice));
    auto bf16_layout = celeg::make_weight_layout(
        celeg::WeightMode::Bf16, dbf16.data(), nullptr);
    bf16_layout->embed_token(1, out.data(), hidden, stream.get());

    std::vector<int8_t> int8_table = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> int8_scales = {1.0f, 0.5f};
    celeg::DeviceBuffer<int8_t> dint8(int8_table.size());
    celeg::DeviceBuffer<float> dint8_scales(int8_scales.size());
    CELEG_CUDA(cudaMemcpy(dint8.data(), int8_table.data(), dint8.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dint8_scales.data(), int8_scales.data(),
                          dint8_scales.bytes(), cudaMemcpyHostToDevice));
    auto int8_layout = celeg::make_weight_layout(
        celeg::WeightMode::Int8, dint8.data(), dint8_scales.data());
    int8_layout->embed_token(1, out.data(), hidden, stream.get());

    std::vector<uint8_t> int4_table = {0x21U, 0x43U, 0x65U, 0x87U};
    std::vector<float> int4_scales = {0.25f, 0.5f};
    celeg::DeviceBuffer<uint8_t> dint4(int4_table.size());
    celeg::DeviceBuffer<float> dint4_scales(int4_scales.size());
    CELEG_CUDA(cudaMemcpy(dint4.data(), int4_table.data(), dint4.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dint4_scales.data(), int4_scales.data(),
                          dint4_scales.bytes(), cudaMemcpyHostToDevice));
    auto int4_layout = celeg::make_weight_layout(
        celeg::WeightMode::Int4, dint4.data(), dint4_scales.data());
    int4_layout->embed_token(1, out.data(), hidden, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
}

{
    std::vector<uint8_t> packed = {0x21U, 0x43U, 0xefU, 0xcdU};
    std::vector<float> scales = {0.5f, 0.25f};
    celeg::DeviceBuffer<uint8_t> table(packed.size());
    celeg::DeviceBuffer<float> ds(scales.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(4);
    CELEG_CUDA(cudaMemcpy(table.data(), packed.data(), table.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_embedding_int4(1, table.data(), ds.data(), output.data(),
                               4, stream.get());
    std::vector<__nv_bfloat16> host(4);
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    expect_near(to_float(host[0]), -0.25f, 0.02f);
    expect_near(to_float(host[1]), -0.50f, 0.02f);
    expect_near(to_float(host[2]), -0.75f, 0.02f);
    expect_near(to_float(host[3]), -1.00f, 0.02f);
}

{
    std::vector<int8_t> table = {1, 2, 3, 4, -1, 2, -3, 4};
    std::vector<float> scales = {0.5f, 2.0f};
    int32_t token = 1;
    celeg::DeviceBuffer<int8_t> dt(table.size());
    celeg::DeviceBuffer<float> ds(scales.size());
    celeg::DeviceBuffer<int32_t> dtoken(1);
    celeg::DeviceBuffer<__nv_bfloat16> out(4);
    CELEG_CUDA(cudaMemcpy(dt.data(), table.data(), dt.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(ds.data(), scales.data(), ds.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dtoken.data(), &token, sizeof(token), cudaMemcpyHostToDevice));
    celeg::launch_embedding_int8_device(dtoken.data(), dt.data(), ds.data(),
                                      out.data(), 4, stream.get());
    std::vector<__nv_bfloat16> result(4);
    CELEG_CUDA(cudaMemcpyAsync(result.data(), out.data(), out.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const std::vector<float> expected = {-2.0f, 4.0f, -6.0f, 8.0f};
    for (int i = 0; i < 4; ++i) expect_near(to_float(result[i]), expected[i]);
}
}

}
