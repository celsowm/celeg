#include "celeg/model/weights/quantization.hpp"
#include "support/assertions.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    const std::vector<float> source = {
        -2.0f, -1.0f, 0.0f, 2.0f,
        0.0f, 0.25f, -0.5f, 1.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
    };
    std::vector<uint16_t> bf16(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        bf16[i] = celeg::float_to_bf16_bits(source[i]);
    }
    const auto pack = celeg::quantize_bf16_rows(
        reinterpret_cast<const std::byte*>(bf16.data()), 3, 4);
    CELEG_TEST_CHECK(pack.values.size() == source.size());
    CELEG_TEST_CHECK(pack.scales.size() == 3);
    CELEG_TEST_CHECK(pack.values[3] == 127);
    CELEG_TEST_CHECK(pack.values[0] == -127);
    CELEG_TEST_CHECK(pack.scales[2] == 1.0f);

    const auto restored = celeg::dequantize_int8_rows(pack);
    for (size_t i = 0; i < source.size(); ++i) {
        const size_t row = i / 4;
        const float tolerance = pack.scales[row] * 0.51f + 1e-6f;
        CELEG_TEST_CHECK(std::abs(restored[i] - source[i]) <= tolerance);
    }

    std::vector<int8_t> values(4 * 4, 0);
    std::vector<float> scales(4, 0.0f);
    celeg::quantize_bf16_rows_into(
        reinterpret_cast<const std::byte*>(bf16.data()), 3, 4,
        values, scales, 1);
    CELEG_TEST_CHECK(scales[0] == 0.0f);
    CELEG_TEST_CHECK(values[4 + 3] == 127);


    {
        const std::vector<float> source4 = {
            -3.0f, -1.0f, 0.0f, 1.0f, 3.0f,
             0.0f,  0.5f, -1.0f, 1.5f, 2.0f,
        };
        std::vector<uint16_t> bits(source4.size());
        for (size_t i = 0; i < source4.size(); ++i) {
            bits[i] = celeg::float_to_bf16_bits(source4[i]);
        }
        const auto int4 = celeg::quantize_bf16_rows_int4(
            reinterpret_cast<const std::byte*>(bits.data()), 2, 5);
        CELEG_TEST_CHECK(int4.values.size() == 6);
        CELEG_TEST_CHECK(int4.scales.size() == 2);
        CELEG_TEST_CHECK(int4.values[0] == 0xe9U);
        CELEG_TEST_CHECK(int4.values[1] == 0x20U);
        CELEG_TEST_CHECK(int4.values[2] == 0x07U);
        const auto restored4 = celeg::dequantize_int4_rows(int4);
        for (size_t i = 0; i < source4.size(); ++i) {
            const size_t row = i / 5;
            const float tolerance = int4.scales[row] * 0.51f + 1e-6f;
            CELEG_TEST_CHECK(std::abs(restored4[i] - source4[i]) <= tolerance);
        }

        std::vector<uint8_t> packed(3 * 3, 0xffU);
        std::vector<float> scales4(3, 0.0f);
        celeg::quantize_bf16_rows_int4_into(
            reinterpret_cast<const std::byte*>(bits.data()), 2, 5,
            packed, scales4, 1);
        CELEG_TEST_CHECK(scales4[0] == 0.0f);
        CELEG_TEST_CHECK(scales4[1] > 0.0f && scales4[2] > 0.0f);
    }

    {
        const uint16_t nan_bits = celeg::float_to_bf16_bits(
            std::numeric_limits<float>::quiet_NaN());
        bool int8_rejected = false;
        bool int4_rejected = false;
        try {
            (void)celeg::quantize_bf16_rows(
                reinterpret_cast<const std::byte*>(&nan_bits), 1, 1);
        } catch (const std::invalid_argument&) {
            int8_rejected = true;
        }
        try {
            (void)celeg::quantize_bf16_rows_int4(
                reinterpret_cast<const std::byte*>(&nan_bits), 1, 1);
        } catch (const std::invalid_argument&) {
            int4_rejected = true;
        }
        CELEG_TEST_CHECK(int8_rejected && int4_rejected);
    }

    std::cout << "quantization_test: ok\n";
    return 0;
}
