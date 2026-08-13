#include "celeg/checkpoint/packed/int4.hpp"
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

} // namespace

int main() {
    MemoryRepository repository;
    // compressed-tensors stores centered INT4 codes: 0..15 represents -8..7.
    const std::vector<uint32_t> packed(2 * 4, 0x0f0f0f0fu);
    const std::vector<uint16_t> scales(2 * 1, 0x3f80u);
    const std::vector<int64_t> shape = {2, 32};
    repository.tensors.emplace("w_packed", view(celeg::TensorDType::I32, {2, 4}, packed));
    repository.tensors.emplace("w_scale", view(celeg::TensorDType::BF16, {2, 1}, scales));
    repository.tensors.emplace("w_shape", view(celeg::TensorDType::I64, {2}, shape));
    CELEG_TEST_CHECK(celeg::has_packed_int4_matrix(repository, "w"));
    const auto matrix = celeg::load_packed_int4_matrix(repository, "w", {2, 32});
    const auto values = celeg::dequantize_packed_int4(matrix);
    CELEG_TEST_CHECK(matrix.rows == 2 && matrix.cols == 32 && matrix.group_size == 32);
    CELEG_TEST_CHECK(values.front() == 7.0f && values[1] == -8.0f);
    CELEG_TEST_CHECK(values[32] == 7.0f && values[33] == -8.0f);
    std::cout << "packed_int4_test: ok\n";
}
