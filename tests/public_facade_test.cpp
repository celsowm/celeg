#include "celeg/model/model.hpp"
#include "support/assertions.hpp"
#include "celeg/runtime/concurrency.hpp"
#include <iostream>
#include <memory>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<celeg::Model>);
static_assert(!std::is_copy_constructible_v<celeg::ConcurrentEngine>);
static_assert(sizeof(celeg::ConcurrentEngine) == sizeof(std::unique_ptr<void>));
static_assert(sizeof(celeg::Model) <= 8 * sizeof(void*));

int main() {
    CELEG_TEST_CHECK(sizeof(celeg::ConcurrentEngine) == sizeof(void*));
    std::cout << "public_facade_test: ok\n";
}
