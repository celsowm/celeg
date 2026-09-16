#include "celeg/checkpoint/packed/fp8.hpp"
#include "celeg/quantization/scalars.hpp"
#include "support/assertions.hpp"

#include <cmath>
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

    /// Absent scale sidecar -> not detected as packed FP8.
    MemoryRepository incomplete;
    incomplete.tensors.emplace("w", view(celeg::TensorDType::F8_E4M3, {2, 4}, raw_e4m3));
    CELEG_TEST_CHECK(!celeg::has_packed_fp8_matrix(incomplete, "w"));

    /// E4M3 anchors verified against torch.float8_e4m3fn: 0x38 is 1.0,
    /// 0x7E is 448.0 (maximum, only M==7 at E==15 is NaN), subnormals
    /// scale as 2^-6 * M/8.
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x00) == 0.0f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x01) == 0.001953125f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x38) == 1.0f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x3C) == 1.5f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x40) == 2.0f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0xB8) == -1.0f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0x7E) == 448.0f);
    CELEG_TEST_CHECK(celeg::e4m3_bits_to_float(0xFE) == -448.0f);
    CELEG_TEST_CHECK(std::isnan(celeg::e4m3_bits_to_float(0x7F)));
    CELEG_TEST_CHECK(std::isnan(celeg::e4m3_bits_to_float(0xFF)));

    /// Dequant multiplies each value by its row scale.
    const auto values = celeg::dequantize_packed_fp8(matrix);
    CELEG_TEST_CHECK(values.size() == 8);
    CELEG_TEST_CHECK(values[0] == 1.0f * 1.5f);
    CELEG_TEST_CHECK(values[1] == 2.0f * 1.5f);
    CELEG_TEST_CHECK(values[4] == celeg::e4m3_bits_to_float(0x01) * 2.0f);

    std::cout << "packed_fp8_test: ok\n";
}
