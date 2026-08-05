#include "celeg/runtime/cache/expert_policy.hpp"
#include "support/assertions.hpp"

#include <stdexcept>

int main() {
    const celeg::ExpertKey first{2, 7};
    const celeg::ExpertKey same{2, 7};
    CELEG_TEST_CHECK(first == same);
    const celeg::ExpertKey packed_key{2, 7};
    CELEG_TEST_CHECK(first.packed() == packed_key.packed());

    celeg::ExpertBudget budget{100, 25};
    CELEG_TEST_CHECK(budget.capacity() == 4);
    budget.payload_bytes = 0;
    bool rejected = false;
    try { budget.validate(); }
    catch (const std::invalid_argument&) { rejected = true; }
    CELEG_TEST_CHECK(rejected);
    CELEG_TEST_CHECK(celeg::expert_priority(2.0, 0) >
                     celeg::expert_priority(1.0, 1));
    return 0;
}
