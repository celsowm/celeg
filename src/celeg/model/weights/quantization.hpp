#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
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

/// IEEE half-precision to float, including subnormals and infinities.
///
/// Kept here beside the BF16 helpers so backends share one implementation;
/// it previously existed as four separate file-local copies. Defined inline
/// rather than out-of-line because the GGUF dot kernels call it once per
/// block inside their hot loops, and those translation units are compiled
/// with ISA-specific target attributes that a cross-TU call would bar the
/// compiler from folding into.
inline float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            // Subnormal: renormalize the mantissa into float's wider exponent.
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 |
                     mantissa << 13;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | mantissa << 13;
    } else {
        result = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

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
