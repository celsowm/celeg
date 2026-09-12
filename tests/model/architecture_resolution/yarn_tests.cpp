#include "yarn_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace celeg::architecture_resolution_test {

void run_yarn_tests(const celeg::ArchitectureCatalog& catalog) {
    auto yarn_metadata = structural_metadata("completely_unknown_yarn_model");
    yarn_metadata.values["rope_scaling.rope_type"] = std::string("yarn");
    yarn_metadata.values["rope_scaling.factor"] = 4.0;
    yarn_metadata.values["rope_scaling.original_max_position_embeddings"] = int64_t(2048);
    yarn_metadata.values["rope_scaling.attention_factor"] = 1.25;
    yarn_metadata.values["rope_scaling.beta_fast"] = 32.0;
    yarn_metadata.values["rope_scaling.beta_slow"] = 1.0;
    celeg::CheckpointView yarn_checkpoint;
    yarn_checkpoint.metadata = std::move(yarn_metadata);
    yarn_checkpoint.repository = std::make_shared<GptxRepository>();
    const auto yarn_model = catalog.select(yarn_checkpoint.metadata).resolve(yarn_checkpoint);
    const auto& yarn_attention =
        std::get<celeg::AttentionSpec>(yarn_model.graph.layers[0].mixer);
    const auto& yarn_position = std::get<celeg::RopePositionSpec>(yarn_attention.position);
    const auto& yarn_scaling = std::get<celeg::YarnRopeScaling>(yarn_position.scaling);
    CELEG_TEST_CHECK(yarn_scaling.factor == 4.0);
    CELEG_TEST_CHECK(yarn_scaling.original_context == 2048);
    CELEG_TEST_CHECK(yarn_scaling.attention_factor == 1.25);
    CELEG_TEST_CHECK(yarn_scaling.beta_fast == 32.0);
    CELEG_TEST_CHECK(yarn_scaling.beta_slow == 1.0);
}

}
