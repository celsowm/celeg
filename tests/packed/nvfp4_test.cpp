#include "celeg/checkpoint/packed/nvfp4.hpp"
#include "celeg/quantization/scalars.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <iostream>
#include <unordered_map>

namespace {

class MemoryRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view name) const override {
        return tensors.contains(std::string(name));
    }
    celeg::HostTensorView tensor(std::string_view name) const override {
        return tensors.at(std::string(name));
    }
    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        for (const auto& [name, _] : tensors) result.push_back(name);
        return result;
    }
    std::unordered_map<std::string, celeg::HostTensorView> tensors;
};

template <typename T>
celeg::HostTensorView view(celeg::TensorDType dtype, std::vector<int64_t> shape,
                           const std::vector<T>& values) {
    return {dtype, std::move(shape), reinterpret_cast<const std::byte*>(values.data()),
            values.size() * sizeof(T)};
}

}

int main() {
    /// rows=2, cols=32 -> packed_cols=16, scale_cols=2 (block size 16).
    MemoryRepository repository;
    std::vector<uint8_t> packed(2 * 16);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(i);
    const std::vector<uint8_t> block_scales = {0x38, 0x40, 0x3A, 0x41};
    const std::vector<float> global_scale = {3.0f};
    repository.tensors.emplace("w_packed", view(celeg::TensorDType::U8, {2, 16}, packed));
    repository.tensors.emplace("w_scale", view(celeg::TensorDType::F8_E4M3, {2, 2}, block_scales));
    repository.tensors.emplace("w_global_scale", view(celeg::TensorDType::F32, {1}, global_scale));
    CELEG_TEST_CHECK(celeg::has_packed_nvfp4_matrix(repository, "w"));
    const auto matrix = celeg::load_packed_nvfp4_matrix(repository, "w", {2, 32});
    CELEG_TEST_CHECK(matrix.rows == 2 && matrix.cols == 32);
    CELEG_TEST_CHECK(matrix.packed == packed);
    CELEG_TEST_CHECK(matrix.block_scales == block_scales);
    CELEG_TEST_CHECK(matrix.global_scale == 3.0f);
    CELEG_TEST_CHECK(matrix.input_global_scale == 1.0f);  /// no sidecar -> default

    const std::vector<float> input_scale = {0.5f};
    repository.tensors.emplace("w.input_global_scale", view(celeg::TensorDType::F32, {1}, input_scale));
    const auto with_input_scale = celeg::load_packed_nvfp4_matrix(repository, "w", {2, 32});
    CELEG_TEST_CHECK(with_input_scale.input_global_scale == 0.5f);

    /// Missing the global-scale sidecar -> not detected as packed NVFP4.
    MemoryRepository incomplete;
    incomplete.tensors.emplace("w_packed", view(celeg::TensorDType::U8, {2, 16}, packed));
    incomplete.tensors.emplace("w_scale", view(celeg::TensorDType::F8_E4M3, {2, 2}, block_scales));
    CELEG_TEST_CHECK(!celeg::has_packed_nvfp4_matrix(incomplete, "w"));

    /// E2M1 nibble anchors: magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6},
    /// 0x8 is the sign bit; matches the device-side decode_e2m1_fallback.
    CELEG_TEST_CHECK(celeg::e2m1_nibble_to_float(0x0) == 0.0f);
    CELEG_TEST_CHECK(celeg::e2m1_nibble_to_float(0x1) == 0.5f);
    CELEG_TEST_CHECK(celeg::e2m1_nibble_to_float(0x7) == 6.0f);
    CELEG_TEST_CHECK(celeg::e2m1_nibble_to_float(0x9) == -0.5f);
    CELEG_TEST_CHECK(celeg::e2m1_nibble_to_float(0xF) == -6.0f);

    /// Dequant is e2m1 * per-16-block e4m3 scale / per-tensor global scale
    /// (input_global_scale is activation-side only and never applies here).
    /// rows=1, cols=32, global_scale=2.0, block scales {1.0, 2.0}.
    MemoryRepository dequant_repo;
    /// col0=0x1(+0.5) col1=0x2(+1.0) | col2=0x2(+1.0) col3=0x9(-0.5) ...
    /// col16=0x3(+1.5) col17=0x4(+2.0).
    std::vector<uint8_t> packed2(16, 0x00);
    packed2[0] = 0x21;
    packed2[1] = 0x92;
    packed2[8] = 0x43;
    const std::vector<uint8_t> scales2 = {0x38, 0x40};
    const std::vector<float> global2 = {2.0f};
    dequant_repo.tensors.emplace("v_packed", view(celeg::TensorDType::U8, {1, 16}, packed2));
    dequant_repo.tensors.emplace("v_scale", view(celeg::TensorDType::F8_E4M3, {1, 2}, scales2));
    dequant_repo.tensors.emplace("v_global_scale", view(celeg::TensorDType::F32, {1}, global2));
    const auto nv = celeg::load_packed_nvfp4_matrix(dequant_repo, "v", {1, 32});
    const auto dequant = celeg::dequantize_packed_nvfp4(nv);
    CELEG_TEST_CHECK(dequant.size() == 32);
    CELEG_TEST_CHECK(dequant[0] == 0.5f * 1.0f / 2.0f);
    CELEG_TEST_CHECK(dequant[1] == 1.0f * 1.0f / 2.0f);
    CELEG_TEST_CHECK(dequant[2] == 1.0f * 1.0f / 2.0f);
    CELEG_TEST_CHECK(dequant[3] == -0.5f * 1.0f / 2.0f);
    CELEG_TEST_CHECK(dequant[16] == 1.5f * 2.0f / 2.0f);
    CELEG_TEST_CHECK(dequant[17] == 2.0f * 2.0f / 2.0f);

    std::cout << "packed_nvfp4_test: ok\n";
}
