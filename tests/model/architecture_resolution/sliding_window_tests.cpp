#include "sliding_window_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>
#include <vector>

namespace celeg::architecture_resolution_test {

void run_sliding_window_tests(const celeg::ArchitectureCatalog& catalog) {
    auto sliding_metadata = structural_metadata("completely_unknown_name");
    sliding_metadata.values["layer_types"] =
        std::vector<std::string>{"sliding_attention"};
    sliding_metadata.values["sliding_window"] = int64_t(4096);
    celeg::CheckpointView sliding_checkpoint;
    sliding_checkpoint.metadata = std::move(sliding_metadata);
    sliding_checkpoint.repository = std::make_shared<GptxRepository>();
    const auto sliding_model = catalog.select(sliding_checkpoint.metadata)
                                   .resolve(sliding_checkpoint);
    const auto& sliding_attention =
        std::get<celeg::AttentionSpec>(sliding_model.graph.layers[0].mixer);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::SlidingWindowPattern>(
        sliding_attention.pattern));
    CELEG_TEST_CHECK(std::get<celeg::SlidingWindowPattern>(sliding_attention.pattern).window ==
                     4096);

    auto truncated_pattern = structural_metadata("unknown_test_model");
    truncated_pattern.values["num_hidden_layers"] = int64_t(2);
    truncated_pattern.values["layer_types"] =
        std::vector<std::string>{"full_attention"};
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(truncated_pattern), celeg::ResolutionFailureKind::IncompleteLayerSchedule));
}

}
