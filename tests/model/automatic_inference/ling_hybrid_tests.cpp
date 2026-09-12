#include "ling_hybrid_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <cstdio>

namespace celeg::automatic_inference_test {

void run_ling_hybrid_tests(const celeg::ArchitectureCatalog& catalog) {
    celeg::CheckpointView ling_checkpoint;
    ling_checkpoint.metadata = ling_metadata();
    ling_checkpoint.repository = ling_repository();
    celeg::ResolvedModel ling_model;
    try {
        ling_model = catalog.select(ling_checkpoint.metadata).resolve(ling_checkpoint);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ling failure: %s\n", error.what());
        throw;
    }
    CELEG_TEST_CHECK(std::holds_alternative<celeg::GatedDeltaNetSpec>(
        ling_model.graph.layers[0].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        ling_model.graph.layers[3].mixer));
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(ling_model.graph.layers[3].mixer)
                         .latent_state()->factorized());
    CELEG_TEST_CHECK(std::holds_alternative<celeg::DenseFeedForwardSpec>(
        ling_model.graph.layers[0].feed_forward));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::MixtureOfExpertsSpec>(
        ling_model.graph.layers[1].feed_forward));
    CELEG_TEST_CHECK(celeg::explain_resolution(ling_checkpoint).failures.empty());
}

}
