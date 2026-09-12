#include "hybrid_gguf_tests.hpp"

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "support.hpp"
#include "support/assertions.hpp"

namespace celeg::automatic_inference_test {

void run_hybrid_gguf_tests(const celeg::ArchitectureCatalog& catalog) {
    celeg::CheckpointView hybrid_checkpoint;
    hybrid_checkpoint.metadata = hybrid_gguf_metadata();
    hybrid_checkpoint.repository = hybrid_gguf_repository();
    const celeg::ResolvedModel hybrid_model =
        catalog.select(hybrid_checkpoint.metadata).resolve(hybrid_checkpoint);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::ShortConvolutionSpec>(
        hybrid_model.graph.layers[0].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AttentionSpec>(
        hybrid_model.graph.layers[1].mixer));
    CELEG_TEST_CHECK(hybrid_model.graph.tied_embeddings);
}

}
