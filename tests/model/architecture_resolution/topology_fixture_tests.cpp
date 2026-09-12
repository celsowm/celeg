#include "topology_fixture_tests.hpp"

#include "celeg/model/program.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <string>
#include <utility>

namespace celeg::architecture_resolution_test {

void run_topology_fixture_tests(const celeg::ArchitectureCatalog& catalog) {
    for (const auto& [model_type, architecture_id] : {
             std::pair<std::string, std::string>{
                 "celeg_topology_fixture_dense", "topology_fixture_dense"},
             std::pair<std::string, std::string>{
                 "celeg_topology_fixture_grouped_moe", "topology_fixture_grouped_moe"}}) {
        auto fixture_metadata = structural_metadata(model_type);
        celeg::CheckpointView fixture_checkpoint;
        fixture_checkpoint.metadata = std::move(fixture_metadata);
        const auto& fixture_architecture = catalog.select(fixture_checkpoint.metadata);
        CELEG_TEST_CHECK(fixture_architecture.id() == architecture_id);
        const auto fixture_model = fixture_architecture.resolve(fixture_checkpoint);
        const auto compiled = celeg::build_model_program(fixture_model);
        CELEG_TEST_CHECK(compiled.layers.size() == 4);
        if (architecture_id == "topology_fixture_dense") {
            CELEG_TEST_CHECK(std::get<celeg::DenseFeedForwardSpec>(
                                 fixture_model.graph.layers[2].feed_forward).intermediate_size == 8);
            CELEG_TEST_CHECK(fixture_model.graph.layers[2].mixer_norm.after.has_value());
            CELEG_TEST_CHECK(std::get<celeg::CompiledDenseFeedForwardProgram>(
                                 compiled.layers[2].feed_forward).intermediate_size == 8);
        } else {
            const auto& moe = std::get<celeg::MixtureOfExpertsSpec>(
                fixture_model.graph.layers[2].feed_forward);
            const auto& grouped = std::get<celeg::MoeGroupedTopKSelectionSpec>(
                moe.selection);
            CELEG_TEST_CHECK(grouped.group_count == 2);
            CELEG_TEST_CHECK(grouped.experts_per_group == 2);
            CELEG_TEST_CHECK(grouped.groups_per_token == 1);
            const auto& compiled_moe = std::get<celeg::MoeLayerProgram>(
                compiled.layers[2].feed_forward);
            CELEG_TEST_CHECK(std::holds_alternative<celeg::MoeGroupedTopKSelectionSpec>(
                compiled_moe.router.selection));
        }
    }
}

}
