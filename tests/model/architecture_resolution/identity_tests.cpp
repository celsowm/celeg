#include "identity_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

namespace celeg::architecture_resolution_test {

void run_identity_tests(const celeg::ArchitectureCatalog& catalog) {
    const auto identity_a = resolve_structural_identity(catalog, "lizzy");
    const auto identity_b = resolve_structural_identity(catalog, "unknown_test_model");
    const auto poisoned_identity = resolve_structural_identity(catalog, "gemma4");
    CELEG_TEST_CHECK(identity_a.graph.fingerprint() == identity_b.graph.fingerprint());
    CELEG_TEST_CHECK(identity_a.graph.fingerprint() == poisoned_identity.graph.fingerprint());
    CELEG_TEST_CHECK(equivalent_weight_plan(identity_a.weight_plan, identity_b.weight_plan));
    CELEG_TEST_CHECK(equivalent_weight_plan(identity_a.weight_plan,
                                            poisoned_identity.weight_plan));
}

}
