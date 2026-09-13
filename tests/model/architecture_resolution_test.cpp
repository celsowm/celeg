#include "architecture_resolution/catalog_tests.hpp"
#include "architecture_resolution/failure_modes_tests.hpp"
#include "architecture_resolution/gguf_spelling_tests.hpp"
#include "architecture_resolution/hybrid_kda_mla_tests.hpp"
#include "architecture_resolution/identity_tests.hpp"
#include "architecture_resolution/norm_layout_tests.hpp"
#include "architecture_resolution/postnorm_evidence_tests.hpp"
#include "architecture_resolution/rope_layer_schedule_tests.hpp"
#include "architecture_resolution/sliding_window_tests.hpp"
#include "architecture_resolution/stated_policy_tests.hpp"
#include "architecture_resolution/structural_tests.hpp"
#include "architecture_resolution/topology_fixture_tests.hpp"
#include "architecture_resolution/yarn_tests.hpp"
#include "celeg/runtime/context.hpp"

#include <iostream>

int main() {
    const auto runtime = celeg::create_builtin_runtime_context();
    const auto& catalog = runtime->architectures();

    celeg::architecture_resolution_test::run_structural_tests(catalog);
    celeg::architecture_resolution_test::run_identity_tests(catalog);
    celeg::architecture_resolution_test::run_norm_layout_tests(catalog);
    celeg::architecture_resolution_test::run_failure_modes_tests(catalog);
    celeg::architecture_resolution_test::run_sliding_window_tests(catalog);
    celeg::architecture_resolution_test::run_yarn_tests(catalog);
    celeg::architecture_resolution_test::run_postnorm_evidence_tests(catalog);
    celeg::architecture_resolution_test::run_rope_layer_schedule_tests(catalog);
    celeg::architecture_resolution_test::run_stated_policy_tests(catalog);
    celeg::architecture_resolution_test::run_gguf_spelling_tests(catalog);
    celeg::architecture_resolution_test::run_late_failure_modes_tests(catalog);
    celeg::architecture_resolution_test::run_topology_fixture_tests(catalog);
    celeg::architecture_resolution_test::run_hybrid_kda_mla_tests(catalog);
    celeg::architecture_resolution_test::run_catalog_tests();

    std::cout << "architecture_resolution_test: ok\n";
}
