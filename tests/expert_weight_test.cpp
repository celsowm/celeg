#include "lfm/weight_loader.hpp"
#include "lfm/safetensors_repo.hpp"
#include "lfm/detail/model_types.hpp"
#include "lfm/cuda_utils.cuh"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

void write_safetensors(const std::filesystem::path& path,
                       const std::string& header,
                       const void* data, size_t data_size) {
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_size = header.size();
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (data_size) {
        out.write(static_cast<const char*>(data),
                  static_cast<std::streamsize>(data_size));
    }
    out.close();
}

} // namespace

int main() {
 try {
    // Synthetic single-file checkpoint with a packed 3D expert tensor and an
    // F32 expert bias. experts=4, rows_per_expert=3, cols=2.
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
    // 12*2 bf16 = 48 bytes; 4 f32 = 16 bytes; total data = 64.
    static_assert(sizeof(uint16_t) == 2, "bf16 is 2 bytes");
    static_assert(sizeof(float) == 4, "f32 is 4 bytes");

    const auto dir = std::filesystem::temp_directory_path() / "lfm_expert_test";
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / "model.safetensors", std::ios::binary);
        const uint64_t hs = header.size();
        out.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(reinterpret_cast<const char*>(expert_values.data()),
                  static_cast<std::streamsize>(expert_values.size() * 2));
        out.write(reinterpret_cast<const char*>(bias_values.data()),
                  static_cast<std::streamsize>(bias_values.size() * 4));
        out.close();
    }

    {
    lfm::SafeTensorRepository repo(dir / "model.safetensors");
    assert(!repo.sharded());
    assert(repo.contains("experts"));
    assert(repo.contains("bias"));

    auto weights = lfm::WeightLoader::acquire((dir / "model.safetensors").string(),
                                             lfm::WeightMode::Bf16);
    lfm::WeightLoader loader(weights, lfm::WeightMode::Bf16);

    const lfm::ExpertLinearWeight* ew =
        loader.load_expert_linear_weight(repo, "experts", experts, rows_per_expert, cols);
    assert(ew != nullptr);
    assert(ew->experts == experts);
    assert(ew->rows_per_expert == rows_per_expert);
    assert(ew->cols == cols);
    assert(ew->kind == lfm::LinearStorageKind::Bf16);
    assert(ew->bf16 != nullptr);

    // expert_view(i) must point at offset i * rows_per_expert * cols.
    for (int e = 0; e < experts; ++e) {
        const lfm::LinearWeight view = ew->expert_view(e);
        assert(view.rows == rows_per_expert);
        assert(view.cols == cols);
        assert(view.bf16 == ew->bf16 + static_cast<size_t>(e) * rows_per_expert * cols);
    }

    // Copy the packed device buffer back and verify exact ordering.
    std::vector<uint16_t> host_expert(total);
    LFM_CUDA(cudaMemcpy(host_expert.data(), ew->bf16, total * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total; ++i) assert(host_expert[i] == expert_values[i]);

    // Invalid expert index must throw.
    bool threw = false;
    try {
        (void)ew->expert_view(experts);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);

    // F32 expert bias.
    const float* bias = loader.load_f32_weight(repo, "bias", {experts});
    assert(bias != nullptr);
    std::vector<float> host_bias(experts);
    LFM_CUDA(cudaMemcpy(host_bias.data(), bias, experts * sizeof(float),
                        cudaMemcpyDeviceToHost));
    for (int i = 0; i < experts; ++i) assert(host_bias[i] == bias_values[i]);

    // Memory accounting must include the BF16 expert buffer and the F32 bias.
    assert(weights->memory_bytes() >= total * sizeof(uint16_t) + experts * sizeof(float));
    }

    std::filesystem::remove_all(dir);
    std::cout << "expert_weight_test: ok\n";
    return 0;
 } catch (const std::exception& e) {
    std::cerr << "expert_weight_test FAILED: " << e.what() << "\n";
    return 1;
 }
}
