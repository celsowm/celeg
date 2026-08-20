#include "celeg/checkpoint/packed/fp8.hpp"
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
    MemoryRepository repository;
    const std::vector<uint8_t> raw_e4m3 = {0x38, 0x40, 0xB8, 0x00, 0x01, 0x02, 0x03, 0x04};
    const std::vector<float> scales = {1.5f, 2.0f};
    repository.tensors.emplace("w", view(celeg::TensorDType::F8_E4M3, {2, 4}, raw_e4m3));
    repository.tensors.emplace("w_scale", view(celeg::TensorDType::F32, {2, 1}, scales));
    CELEG_TEST_CHECK(celeg::has_packed_fp8_matrix(repository, "w"));
    const auto matrix = celeg::load_packed_fp8_matrix(repository, "w", {2, 4});
    CELEG_TEST_CHECK(matrix.rows == 2 && matrix.cols == 4);
    CELEG_TEST_CHECK(matrix.values == raw_e4m3);
    CELEG_TEST_CHECK(matrix.scales == scales);

    // Absent scale sidecar -> not detected as packed FP8.
    MemoryRepository incomplete;
    incomplete.tensors.emplace("w", view(celeg::TensorDType::F8_E4M3, {2, 4}, raw_e4m3));
    CELEG_TEST_CHECK(!celeg::has_packed_fp8_matrix(incomplete, "w"));

    std::cout << "packed_fp8_test: ok\n";
}
