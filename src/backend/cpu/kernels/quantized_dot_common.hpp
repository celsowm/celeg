#pragma once

#include <cstddef>
#include <cstdint>

namespace celeg::detail {

inline constexpr int kQ4UnsignedBias = 8;

inline constexpr int decode_signed_q4_nibble(uint8_t nibble) {
    const uint8_t value = nibble & 0x0fU;
    return value >= kQ4UnsignedBias
        ? static_cast<int>(value) - 16
        : static_cast<int>(value);
}

inline int decode_signed_q4(const uint8_t* packed, size_t col) {
    const uint8_t byte = packed[col >> 1];
    const uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return decode_signed_q4_nibble(nibble);
}

static_assert(decode_signed_q4_nibble(0x0) == 0);
static_assert(decode_signed_q4_nibble(0x7) == 7);
static_assert(decode_signed_q4_nibble(0x8) == -8);
static_assert(decode_signed_q4_nibble(0xf) == -1);

}
