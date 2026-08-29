#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/detail/binary_codec.hpp"
#include "support/assertions.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/backend/cuda/model/detail/expert_weights.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

uint16_t f32_to_bf16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>(bits >> 16);
}

void write_safetensors(const std::filesystem::path& path,
                       const std::string& header,
                       const void* data, size_t data_size) {
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_size = header.size();
    celeg::binary::write_le(out, header_size);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (data_size) {
        out.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(data_size));
    }
    out.close();
}

}

int main() {
 try {
    const int experts = 4;
    const int rows_per_expert = 3;
    const int cols = 2;
    const size_t total = static_cast<size_t>(experts) * rows_per_expert * cols;
    std::vector<uint16_t> expert_values(total);
    for (size_t i = 0; i < total; ++i) expert_values[i] = f32_to_bf16(static_cast<float>(i));

    std::vector<float> bias_values(experts);
    for (int i = 0; i < experts; ++i) bias_values[i] = static_cast<float>(100 + i);

    const std::string header =
        R"({"experts":{"dtype":"BF16","shape":[12,2],"data_offsets":[0,48]},)"
        R"("bias":{"dtype":"F32","shape":[4],"data_offsets":[48,64]}})";
    static_assert(sizeof(uint16_t) == 2, "bf16 is 2 bytes");
    static_assert(sizeof(float) == 4, "f32 is 4 bytes");

    const auto dir = std::filesystem::temp_directory_path() / "celeg_expert_test";
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / "model.safetensors", std::ios::binary);
        const uint64_t hs = header.size();
        celeg::binary::write_le(out, hs);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(reinterpret_cast<const char*>(expert_values.data()),
                  static_cast<std::streamsize>(expert_values.size() * 2));
        out.write(reinterpret_cast<const char*>(bias_values.data()),
                  static_cast<std::streamsize>(bias_values.size() * 4));
        out.close();
    }

    {
    celeg::SafeTensorRepository repo(dir / "model.safetensors");
    CELEG_TEST_CHECK(!repo.sharded());
    CELEG_TEST_CHECK(repo.contains("experts"));
    CELEG_TEST_CHECK(repo.contains("bias"));

    auto weights = celeg::WeightLoader::acquire((dir / "model.safetensors").string(),
                                             celeg::WeightMode::Bf16, "test");
    celeg::WeightLoader loader(weights, celeg::WeightMode::Bf16);

    const celeg::ExpertLinearWeight* ew =
        loader.load_expert_linear_weight(repo, "experts", experts, rows_per_expert, cols);
    CELEG_TEST_CHECK(ew != nullptr);
    CELEG_TEST_CHECK(ew->experts == experts);
    CELEG_TEST_CHECK(ew->rows_per_expert == rows_per_expert);
    CELEG_TEST_CHECK(ew->cols == cols);
    CELEG_TEST_CHECK(ew->kind == celeg::ExpertStorageKind::Bf16);
    CELEG_TEST_CHECK(ew->bf16 != nullptr);

    for (int e = 0; e < experts; ++e) {
        const celeg::LinearWeight view = ew->expert_view(e);
        CELEG_TEST_CHECK(view.rows == rows_per_expert);
        CELEG_TEST_CHECK(view.cols == cols);
        CELEG_TEST_CHECK(std::get<celeg::Bf16LinearStorage>(view.storage).data ==
                          ew->bf16 + static_cast<size_t>(e) * rows_per_expert * cols);
    }

    std::vector<uint16_t> host_expert(total);
    CELEG_CUDA(cudaMemcpy(host_expert.data(), ew->bf16, total * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total; ++i) CELEG_TEST_CHECK(host_expert[i] == expert_values[i]);

    bool threw = false;
    try {
        (void)ew->expert_view(experts);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);

    const float* bias = loader.load_f32_weight(repo, "bias", {experts});
    CELEG_TEST_CHECK(bias != nullptr);
    std::vector<float> host_bias(experts);
    CELEG_CUDA(cudaMemcpy(host_bias.data(), bias, experts * sizeof(float),
                        cudaMemcpyDeviceToHost));
    for (int i = 0; i < experts; ++i) CELEG_TEST_CHECK(host_bias[i] == bias_values[i]);

    CELEG_TEST_CHECK(weights->memory_bytes() >= total * sizeof(uint16_t) + experts * sizeof(float));
    }

    std::filesystem::remove_all(dir);
    std::cout << "expert_weight_test: ok\n";
    return 0;
 } catch (const std::exception& e) {
    std::cerr << "expert_weight_test FAILED: " << e.what() << "\n";
    return 1;
 }
}
