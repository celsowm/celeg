#pragma once

#include <stdexcept>
#include <string>

namespace lfm::test {

inline void require(bool condition, const char* expression,
                    const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" +
                                 std::to_string(line) + ": check failed: " +
                                 expression);
    }
}

} // namespace lfm::test

#define LFM_TEST_CHECK(condition) \
    ::lfm::test::require(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
