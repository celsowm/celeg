#include "lfm/model/model.hpp"
#include "support/assertions.hpp"
#include "lfm/runtime/concurrency.hpp"
#include <iostream>
#include <memory>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<lfm::Model>);
static_assert(!std::is_copy_constructible_v<lfm::ConcurrentEngine>);
static_assert(sizeof(lfm::ConcurrentEngine) == sizeof(std::unique_ptr<void>));
static_assert(sizeof(lfm::Model) <= 8 * sizeof(void*));

int main() {
    LFM_TEST_CHECK(sizeof(lfm::ConcurrentEngine) == sizeof(void*));
    std::cout << "public_facade_test: ok\n";
}
