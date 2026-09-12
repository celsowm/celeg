#include "hf_fixtures_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <cmath>

namespace celeg::automatic_inference_test {

void run_hf_fixtures_tests(const celeg::ArchitectureCatalog& catalog) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata();
    checkpoint.repository = repository();

    const auto& architecture = catalog.select(checkpoint.metadata);
    const celeg::ResolvedModel model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.provenance.identity.find("automatic") != std::string::npos);
    CELEG_TEST_CHECK(model.graph.hidden == 8);
    CELEG_TEST_CHECK(model.graph.embedding_transform.multiplier == 2.0f);
    CELEG_TEST_CHECK(std::abs(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).query_scale -
                              0.3535533906f) < 1.0e-6f);
    CELEG_TEST_CHECK(model.graph.layers[0].residual.multiplier == 0.5f);
    CELEG_TEST_CHECK(model.graph.logits_divisor == 2.0f);
    CELEG_TEST_CHECK(model.graph.layers.size() == 2);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        model.graph.layers.front().mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::OrthogonalizeCurrentValueSpec>(
        std::get<celeg::AttentionSpec>(model.graph.layers.front().mixer).output_transform));
    CELEG_TEST_CHECK(celeg::explain_resolution(checkpoint).failures.empty());
}

}
