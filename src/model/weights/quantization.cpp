#include "celeg/model/weights/quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace celeg {

namespace {

size_t checked_product(size_t a, size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " dimensions overflow");
    }
    return a * b;
}

size_t checked_sum(size_t a, size_t b, const char* label) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(std::string(label) + " dimensions overflow");
    }
    return a + b;
}

float read_finite_bf16(const std::byte* data, size_t index) {
    uint16_t bits = 0;
    std::memcpy(&bits, data + index * sizeof(uint16_t), sizeof(bits));
    const float value = bf16_bits_to_float(bits);
    if (!std::isfinite(value)) {
        throw std::invalid_argument("quantization source contains non-finite BF16 value");
    }
    return value;
}

} // namespace

float bf16_bits_to_float(uint16_t bits) {
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

uint16_t float_to_bf16_bits(float value) {
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    // Round-to-nearest-even before truncating the lower 16 bits.
    const uint32_t lsb = (word >> 16) & 1U;
    word += 0x7FFFU + lsb;
    return static_cast<uint16_t>(word >> 16);
}

void quantize_bf16_rows_into(const std::byte* data,
                             size_t rows,
                             size_t cols,
                             std::vector<int8_t>& values,
                             std::vector<float>& scales,
                             size_t row_offset) {
    const size_t element_count = checked_product(rows, cols, "INT8 quantization");
    const size_t end_row = checked_sum(row_offset, rows, "INT8 quantization");
    const size_t required_values = checked_product(end_row, cols, "INT8 quantization");
    if (!data && element_count != 0) {
        throw std::invalid_argument("quantize_bf16_rows_into received null data");
    }
    if (values.size() < required_values || scales.size() < end_row) {
        throw std::invalid_argument("quantization destination is too small");
    }
    for (size_t row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = read_finite_bf16(data, row * cols + col);
            maximum = std::max(maximum, std::abs(value));
        }
        const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scales[row_offset + row] = scale;
        for (size_t col = 0; col < cols; ++col) {
            const float normalized =
                read_finite_bf16(data, row * cols + col) / scale;
            const int rounded = static_cast<int>(std::nearbyint(normalized));
            values[(row_offset + row) * cols + col] =
                static_cast<int8_t>(std::clamp(rounded, -127, 127));
        }
    }
}

Int8RowwisePack quantize_bf16_rows(const std::byte* data,
                                   size_t rows,
                                   size_t cols) {
    Int8RowwisePack pack;
    pack.rows = rows;
    pack.cols = cols;
    pack.values.resize(checked_product(rows, cols, "INT8 quantization"));
    pack.scales.resize(rows);
    quantize_bf16_rows_into(data, rows, cols, pack.values, pack.scales, 0);
    return pack;
}

std::vector<float> dequantize_int8_rows(const Int8RowwisePack& pack) {
    if (pack.values.size() != pack.rows * pack.cols ||
        pack.scales.size() != pack.rows) {
        throw std::invalid_argument("invalid Int8RowwisePack dimensions");
    }
    std::vector<float> output(pack.values.size());
    for (size_t row = 0; row < pack.rows; ++row) {
        for (size_t col = 0; col < pack.cols; ++col) {
            const size_t index = row * pack.cols + col;
            output[index] = static_cast<float>(pack.values[index]) *
                            pack.scales[row];
        }
    }
    return output;
}


namespace {

uint8_t encode_int4(int value) {
    return static_cast<uint8_t>(value) & 0x0fU;
}

int decode_int4(uint8_t nibble) {
    nibble &= 0x0fU;
    return nibble >= 8U ? static_cast<int>(nibble) - 16
                        : static_cast<int>(nibble);
}

} // namespace

void quantize_bf16_rows_int4_into(const std::byte* data,
                                  size_t rows,
                                  size_t cols,
                                  std::vector<uint8_t>& values,
                                  std::vector<float>& scales,
                                  size_t row_offset) {
    const size_t element_count = checked_product(rows, cols, "INT4 quantization");
    const size_t packed_cols = cols / 2 + cols % 2;
    const size_t end_row = checked_sum(row_offset, rows, "INT4 quantization");
    const size_t required_values =
        checked_product(end_row, packed_cols, "INT4 quantization");
    if (!data && element_count != 0) {
        throw std::invalid_argument("quantize_bf16_rows_int4_into received null data");
    }
    if (values.size() < required_values || scales.size() < end_row) {
        throw std::invalid_argument("INT4 quantization destination is too small");
    }
    for (size_t row = 0; row < rows; ++row) {
        float maximum = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = read_finite_bf16(data, row * cols + col);
            maximum = std::max(maximum, std::abs(value));
        }
        const float scale = maximum > 0.0f ? maximum / 7.0f : 1.0f;
        scales[row_offset + row] = scale;
        uint8_t* destination = values.data() + (row_offset + row) * packed_cols;
        std::fill(destination, destination + packed_cols, uint8_t{0});
        for (size_t col = 0; col < cols; ++col) {
            const float normalized =
                read_finite_bf16(data, row * cols + col) / scale;
            const int rounded = std::clamp(
                static_cast<int>(std::nearbyint(normalized)), -7, 7);
            const uint8_t nibble = encode_int4(rounded);
            uint8_t& packed = destination[col / 2];
            if ((col & 1U) == 0U) {
                packed = static_cast<uint8_t>((packed & 0xf0U) | nibble);
            } else {
                packed = static_cast<uint8_t>((packed & 0x0fU) | (nibble << 4));
            }
        }
    }
}

Int4RowwisePack quantize_bf16_rows_int4(const std::byte* data,
                                        size_t rows,
                                        size_t cols) {
    Int4RowwisePack pack;
    pack.rows = rows;
    pack.cols = cols;
    const size_t packed_cols = cols / 2 + cols % 2;
    pack.values.resize(checked_product(rows, packed_cols, "INT4 quantization"));
    pack.scales.resize(rows);
    quantize_bf16_rows_int4_into(data, rows, cols, pack.values, pack.scales, 0);
    return pack;
}

std::vector<float> dequantize_int4_rows(const Int4RowwisePack& pack) {
    const size_t packed_cols = pack.packed_cols();
    if (pack.values.size() != pack.rows * packed_cols ||
        pack.scales.size() != pack.rows) {
        throw std::invalid_argument("invalid Int4RowwisePack dimensions");
    }
    std::vector<float> output(
        checked_product(pack.rows, pack.cols, "INT4 dequantization"));
    for (size_t row = 0; row < pack.rows; ++row) {
        const uint8_t* packed = pack.values.data() + row * packed_cols;
        for (size_t col = 0; col < pack.cols; ++col) {
            const uint8_t byte = packed[col / 2];
            const uint8_t nibble = (col & 1U) == 0U ? byte & 0x0fU : byte >> 4;
            output[row * pack.cols + col] =
                static_cast<float>(decode_int4(nibble)) * pack.scales[row];
        }
    }
    return output;
}

} // namespace celeg
