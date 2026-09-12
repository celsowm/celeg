#include "structural_tests.hpp"

#include "celeg/model/program.hpp"
#include "celeg/runtime/context.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace celeg::architecture_resolution_test {

void run_structural_tests(const celeg::ArchitectureCatalog& catalog) {
    CELEG_TEST_CHECK(catalog.find("automatic") != nullptr);

    for (const std::string model_type : {"lfm2", "qwen3_5", "granite", "gemma4"}) {
        const auto metadata = structural_metadata(model_type);
        CELEG_TEST_CHECK(catalog.select(metadata).id() == "automatic");
    }

    auto metadata = structural_metadata("gptx2");
    metadata.repository_hint = "AxiomicLabs/GPT-X2.5-135M";
    metadata.values["num_hidden_layers"] = int64_t(1);
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = metadata;
    checkpoint.repository = std::make_shared<GptxRepository>();
    const auto& architecture = catalog.select(metadata);
    const auto model = architecture.resolve(checkpoint);
    CELEG_TEST_CHECK(model.provenance.architecture_id == "automatic");
    CELEG_TEST_CHECK(model.graph.hidden == 576);
    CELEG_TEST_CHECK(model.graph.layers.size() == 1);
    CELEG_TEST_CHECK(model.graph.tied_embeddings);
    CELEG_TEST_CHECK(std::get<celeg::AttentionSpec>(model.graph.layers[0].mixer).query_scale ==
                     0.125f);
}

}
