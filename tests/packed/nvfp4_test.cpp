#include "celeg/checkpoint/packed/nvfp4.hpp"
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
    // rows=2, cols=32 -> packed_cols=16, scale_cols=2 (block size 16).
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
    CELEG_TEST_CHECK(matrix.input_global_scale == 1.0f);  // no sidecar -> default

    const std::vector<float> input_scale = {0.5f};
    repository.tensors.emplace("w.input_global_scale", view(celeg::TensorDType::F32, {1}, input_scale));
    const auto with_input_scale = celeg::load_packed_nvfp4_matrix(repository, "w", {2, 32});
    CELEG_TEST_CHECK(with_input_scale.input_global_scale == 0.5f);

    // Missing the global-scale sidecar -> not detected as packed NVFP4.
    MemoryRepository incomplete;
    incomplete.tensors.emplace("w_packed", view(celeg::TensorDType::U8, {2, 16}, packed));
    incomplete.tensors.emplace("w_scale", view(celeg::TensorDType::F8_E4M3, {2, 2}, block_scales));
    CELEG_TEST_CHECK(!celeg::has_packed_nvfp4_matrix(incomplete, "w"));

    std::cout << "packed_nvfp4_test: ok\n";
}
