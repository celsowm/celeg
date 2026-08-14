#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace celeg {

inline size_t checked_multiply(size_t a, size_t b, const char* message) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error(message);
    }
    return a * b;
}

inline size_t checked_add(size_t a, size_t b, const char* message) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::overflow_error(message);
    }
    return a + b;
}

inline size_t checked_round_up(size_t value, size_t alignment,
                               const char* message) {
    if (alignment == 0) {
        throw std::invalid_argument("checked_round_up alignment must be positive");
    }
    return checked_multiply(
        checked_add(value, alignment - 1, message) / alignment,
        alignment, message);
}

} // namespace celeg
