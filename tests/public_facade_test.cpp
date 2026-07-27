#include "lfm/model/model.hpp"
#include "lfm/runtime/concurrency.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<lfm::LfmModel>);
static_assert(!std::is_copy_constructible_v<lfm::ConcurrentEngine>);
static_assert(sizeof(lfm::ConcurrentEngine) == sizeof(std::unique_ptr<void>));
static_assert(sizeof(lfm::LfmModel) <= 8 * sizeof(void*));

int main() {
    assert(sizeof(lfm::ConcurrentEngine) == sizeof(void*));
    std::cout << "public_facade_test: ok\n";
}
