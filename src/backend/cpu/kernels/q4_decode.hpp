#pragma once

#include <cstddef>
#include <cstdint>

namespace celeg::detail {

inline int decode_q4_signed_nibble(const std::uint8_t* packed, std::size_t col) {
    const std::uint8_t byte = packed[col >> 1];
    const std::uint8_t nibble = (col & 1U) == 0 ? byte & 0x0fU : byte >> 4;
    return nibble >= 8U ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
}

}
