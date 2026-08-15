#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

struct Int8RowwisePack {
    std::vector<int8_t> values;
    std::vector<float> scales;
    size_t rows = 0;
    size_t cols = 0;
};


struct Int4RowwisePack {
    std::vector<uint8_t> values;
    std::vector<float> scales;
    size_t rows = 0;
    size_t cols = 0;

    size_t packed_cols() const { return (cols + 1) / 2; }
};

float bf16_bits_to_float(uint16_t bits);
uint16_t float_to_bf16_bits(float value);

Int8RowwisePack quantize_bf16_rows(const std::byte* data,
                                   size_t rows,
                                   size_t cols);

void quantize_bf16_rows_into(const std::byte* data,
                             size_t rows,
                             size_t cols,
                             std::vector<int8_t>& values,
                             std::vector<float>& scales,
                             size_t row_offset);

std::vector<float> dequantize_int8_rows(const Int8RowwisePack& pack);

Int4RowwisePack quantize_bf16_rows_int4(const std::byte* data,
                                        size_t rows,
                                        size_t cols);

void quantize_bf16_rows_int4_into(const std::byte* data,
                                  size_t rows,
                                  size_t cols,
                                  std::vector<uint8_t>& values,
                                  std::vector<float>& scales,
                                  size_t row_offset);

std::vector<float> dequantize_int4_rows(const Int4RowwisePack& pack);

}
