#include "celeg/checkpoint/packed/int8.hpp"
#include "support/assertions.hpp"

#include <cstring>
#include <iostream>
#include <unordered_map>

namespace {

class MemoryRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view name) const override { return tensors.contains(std::string(name)); }
    celeg::HostTensorView tensor(std::string_view name) const override { return tensors.at(std::string(name)); }
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
    const std::vector<uint32_t> packed = {
        0x7F80FF00u, 0x00000081u,
        0xFFFE7E00u, 0x00000080u};
    const std::vector<uint16_t> scales = {0x3f80u, 0x4000u};
    const std::vector<int64_t> shape = {2, 5};
    repository.tensors.emplace("w_packed", view(celeg::TensorDType::I32, {2, 2}, packed));
    repository.tensors.emplace("w_scale", view(celeg::TensorDType::BF16, {2, 1}, scales));
    repository.tensors.emplace("w_shape", view(celeg::TensorDType::I64, {2}, shape));
    CELEG_TEST_CHECK(celeg::has_packed_int8_matrix(repository, "w"));
    const auto matrix = celeg::load_packed_int8_matrix(repository, "w", {2, 5});
    const std::vector<int8_t> expected_values = {-128, 127, 0, -1, 1,
                                                 -128, -2, 126, 127, 0};
    const std::vector<float> expected_scales = {1.0f, 2.0f};
    CELEG_TEST_CHECK(matrix.rows == 2 && matrix.cols == 5);
    CELEG_TEST_CHECK(matrix.values == expected_values);
    CELEG_TEST_CHECK(matrix.scales == expected_scales);
    std::cout << "packed_int8_test: ok\n";
}
