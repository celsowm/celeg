#pragma once

#include <cstdint>
#include <cstring>

namespace celeg {

inline float bf16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

inline std::uint16_t float_to_bf16_bits(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    const std::uint32_t lsb = (word >> 16) & 1U;
    word += 0x7FFFU + lsb;
    return static_cast<std::uint16_t>(word >> 16);
}

inline float fp16_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    std::uint32_t exponent = (bits >> 10) & 0x1fu;
    std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign | static_cast<std::uint32_t>(127 - 14 - shift) << 23 |
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

}
