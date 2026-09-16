#pragma once

#include <cmath>
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

inline float fp16_bits_to_float(std::uint16_t bits) {    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
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

/// Encodes a float as IEEE-754 binary16 with round-to-nearest-even. This
/// is the counterpart of fp16_bits_to_float for host-side ggml packing
/// (Q4_K d/dmin); out-of-range magnitudes saturate to inf, NaN stays NaN.
inline std::uint16_t float_to_fp16_bits(float value) {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    const std::uint16_t sign = static_cast<std::uint16_t>((word >> 16) & 0x8000u);
    const int exponent = static_cast<int>((word >> 23) & 0xff);
    const std::uint32_t mantissa = word & 0x007fffffu;
    if (exponent == 0xff) {
        return static_cast<std::uint16_t>(sign | 0x7c00u |
                                          (mantissa != 0 ? 0x0200u : 0u));
    }
    const int half_exponent = exponent - 112;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exponent < 1) {
        /// Subnormal output: round the f32 value to an integer count of
        /// 2^-24 quanta with ties-to-even; beyond that the value rounds
        /// to signed zero.
        const int shift = 126 - exponent;
        if (shift > 23) return sign;
        const std::uint32_t mant = mantissa | 0x00800000u;
        const std::uint32_t half_quantum = 1u << (shift - 1);
        const std::uint32_t out =
            (mant + half_quantum - 1 + ((mant >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | out);
    }
    const std::uint32_t lsb = (mantissa >> 13) & 1u;
    const std::uint32_t rounded = mantissa + 0x00000fffu + lsb;
    int result_exponent = half_exponent;
    if (rounded & 0x00800000u) {
        ++result_exponent;
        if (result_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(sign | (result_exponent << 10) |
                                      (rounded >> 13));
}

/// Decodes one OCP FP8 E4M3 (finite, no infinities) bit pattern to float.
/// Exponent bias is 7: 0x38 is 1.0, 0x7E is 448.0 (maximum), 0x7F/0xFF are
/// NaN. Anchor values verified against torch.float8_e4m3fn.
inline float e4m3_bits_to_float(std::uint8_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x80u) << 24;
    const std::uint32_t exponent = (bits >> 3) & 0x0fu;
    const std::uint32_t mantissa = bits & 0x07u;
    std::uint32_t result = 0;
    if (exponent == 0) {
        /// Subnormals have no implicit leading one: 2^-6 * M/8 == ldexp(M, -9).
        float value = std::ldexp(static_cast<float>(mantissa), -9);
        if (sign != 0) value = -value;
        return value;
    } else if (exponent == 15 && mantissa == 7) {
        result = sign | 0x7f800000u | mantissa << 20;
    } else {
        result = sign | (exponent + (127 - 7)) << 23 | mantissa << 20;
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

/// Decodes one OCP MX E2M1 (NVFP4) nibble to float. Magnitudes are
/// {0, 0.5, 1, 1.5, 2, 3, 4, 6} with 0x8 as the sign bit; there are no
/// NaN or infinite codes. Matches the device-side decode_e2m1_fallback
/// LUT in src/backend/cuda/kernels/linear.cuh.
inline float e2m1_nibble_to_float(std::uint8_t nibble) {
    constexpr float kMagnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                     2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = kMagnitude[nibble & 0x7u];
    return (nibble & 0x8u) != 0u ? -magnitude : magnitude;
}

}
