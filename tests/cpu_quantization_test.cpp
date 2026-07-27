#include "lfm/backend/cpu/quantization.hpp"
#include "support/assertions.hpp"
#include "lfm/model/weights/quantization.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
    constexpr size_t rows = 3, cols = 67;
    std::vector<float> values(rows * cols);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i) * 0.17f) * 2.0f;
    const auto pack = lfm::quantize_float_groupwise_q4(values.data(), rows, cols, 32);
    pack.validate();
    std::vector<float> row(cols);
    float max_error = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        lfm::dequantize_q4_row(pack, r, row.data());
        for (size_t c = 0; c < cols; ++c) {
            max_error = std::max(max_error, std::abs(row[c] - values[r * cols + c]));
        }
    }
    LFM_TEST_CHECK(max_error < 0.31f);

    const auto q8 = lfm::quantize_float_groupwise_q8(values.data(), cols, 32);
    q8.validate();
    float max_q8_error = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        const size_t group = i / q8.group_size;
        const float reconstructed = static_cast<float>(q8.values[i]) * q8.scales[group];
        max_q8_error = std::max(max_q8_error, std::abs(reconstructed - values[i]));
    }
    LFM_TEST_CHECK(max_q8_error < 0.02f);

    std::vector<uint16_t> bf16(7);
    for (size_t i = 0; i < bf16.size(); ++i) bf16[i] = lfm::float_to_bf16_bits(static_cast<float>(i) / 3.0f);
    const auto path = std::filesystem::temp_directory_path() / "lfm25-cpu-pack-test.bin";
    {
        lfm::CpuPackMetadata metadata;
        metadata.source_id = "unit-test";
        metadata.isa = "scalar";
        metadata.group_size = 32;
        lfm::CpuPackWriter writer(path, metadata);
        writer.add_q4_matrix("matrix", pack);
        writer.add_bf16_vector("vector", reinterpret_cast<const std::byte*>(bf16.data()), bf16.size());
        writer.commit();
    }
    {
        lfm::CpuPackReader reader(path);
        LFM_TEST_CHECK(reader.metadata().source_id == "unit-test");
        const auto loaded = reader.read_q4_matrix("matrix");
        LFM_TEST_CHECK(loaded.values == pack.values);
        LFM_TEST_CHECK(loaded.scales_bf16 == pack.scales_bf16);
        const auto vector = reader.read_bf16_vector("vector");
        LFM_TEST_CHECK(vector.size() == bf16.size());
        for (size_t i = 0; i < vector.size(); ++i) LFM_TEST_CHECK(std::abs(vector[i] - lfm::bf16_bits_to_float(bf16[i])) < 1e-6f);
    }
    std::filesystem::remove(path);
    std::cout << "cpu_quantization_test: max_q4_error=" << max_error
              << " max_q8_error=" << max_q8_error << '\n';
}
