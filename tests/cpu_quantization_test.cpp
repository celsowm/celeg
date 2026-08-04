#include "celeg/backend/cpu/quantization.hpp"
#include "support/assertions.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

int main() {
    constexpr size_t rows = 3, cols = 67;
    std::vector<float> values(rows * cols);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.17f) * 2.0f;
    }
    const auto pack = celeg::quantize_float_groupwise_q4(
        values.data(), rows, cols, 32);
    pack.validate();
    std::vector<float> row(cols);
    float max_error = 0.0f;
    for (size_t r = 0; r < rows; ++r) {
        celeg::dequantize_q4_row(pack, r, row.data());
        for (size_t c = 0; c < cols; ++c) {
            max_error = std::max(
                max_error, std::abs(row[c] - values[r * cols + c]));
        }
    }
    CELEG_TEST_CHECK(max_error < 0.31f);

    const auto q8 = celeg::quantize_float_groupwise_q8(
        values.data(), cols, 32);
    q8.validate();
    float max_q8_error = 0.0f;
    for (size_t i = 0; i < cols; ++i) {
        const size_t group = i / q8.group_size;
        const float reconstructed =
            static_cast<float>(q8.values[i]) * q8.scales[group];
        max_q8_error = std::max(
            max_q8_error, std::abs(reconstructed - values[i]));
    }
    CELEG_TEST_CHECK(max_q8_error < 0.02f);

    std::vector<uint16_t> bf16(7);
    for (size_t i = 0; i < bf16.size(); ++i) {
        bf16[i] = celeg::float_to_bf16_bits(static_cast<float>(i) / 3.0f);
    }
    const auto path =
        std::filesystem::temp_directory_path() / "celeg-cpu-pack-test.bin";
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    std::filesystem::remove(path);
    std::filesystem::remove(temporary);
    {
        celeg::CpuPackMetadata metadata;
        metadata.source_id = "unit-test";
        metadata.isa = "scalar";
        metadata.group_size = 32;
        celeg::CpuPackWriter writer(path, metadata);
        CELEG_TEST_CHECK(std::filesystem::exists(temporary));
        writer.add_q4_matrix("matrix", pack);
        writer.add_bf16_vector(
            "vector", reinterpret_cast<const std::byte*>(bf16.data()),
            bf16.size());
        writer.commit();
        CELEG_TEST_CHECK(std::filesystem::exists(path));
        CELEG_TEST_CHECK(!std::filesystem::exists(temporary));
    }
    {
        celeg::CpuPackReader reader(path);
        CELEG_TEST_CHECK(reader.metadata().source_id == "unit-test");
        CELEG_TEST_CHECK(reader.contains("matrix"));
        CELEG_TEST_CHECK(reader.contains("vector"));

        const auto loaded = reader.read_q4_matrix("matrix");
        CELEG_TEST_CHECK(loaded.values == pack.values);
        CELEG_TEST_CHECK(loaded.scales_bf16 == pack.scales_bf16);

        std::atomic<int> concurrent_failures{0};
        std::vector<std::thread> threads;
        for (int index = 0; index < 8; ++index) {
            threads.emplace_back([&]() {
                try {
                    const auto concurrent = reader.read_q4_matrix("matrix");
                    if (concurrent.values != pack.values ||
                        concurrent.scales_bf16 != pack.scales_bf16) {
                        ++concurrent_failures;
                    }
                } catch (...) {
                    ++concurrent_failures;
                }
            });
        }
        for (std::thread& thread : threads) thread.join();
        CELEG_TEST_CHECK(concurrent_failures == 0);

        const auto vector = reader.read_bf16_vector("vector");
        CELEG_TEST_CHECK(vector.size() == bf16.size());
        for (size_t i = 0; i < vector.size(); ++i) {
            CELEG_TEST_CHECK(
                std::abs(vector[i] - celeg::bf16_bits_to_float(bf16[i])) <
                1e-6f);
        }
    }
    std::filesystem::remove(path);
    std::cout << "cpu_quantization_test: max_q4_error=" << max_error
              << " max_q8_error=" << max_q8_error << '\n';
}
