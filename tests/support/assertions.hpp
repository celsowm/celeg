#pragma once

#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && std::getenv(name) != nullptr) return 0;
    return ::_putenv_s(name, value);
}

inline int unsetenv(const char* name) {
    return ::_putenv_s(name, "");
}
#endif

namespace celeg::test {

inline void require(bool condition, const char* expression,
                    const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expression);
        throw std::runtime_error(std::string(file) + ":" +
                                 std::to_string(line) + ": check failed: " +
                                 expression);
    }
}

}

#define CELEG_TEST_CHECK(condition) \
    ::celeg::test::require(static_cast<bool>(condition), #condition, __FILE__, __LINE__)
