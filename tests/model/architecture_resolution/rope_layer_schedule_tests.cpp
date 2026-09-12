#include "rope_layer_schedule_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace celeg::architecture_resolution_test {

void run_rope_layer_schedule_tests(const celeg::ArchitectureCatalog& catalog) {
    /// `rope_layer_flags`: a false entry disables rotary position on exactly
    /// that layer (the Lizzy spelling), while true entries keep the resolved
    /// schedule. A ragged array fails loudly instead of misaligning layers.
    {
        celeg::CheckpointMetadata flagged = postnorm_evidence_metadata("flagged_rope_model");
        flagged.values["rope_layer_flags"] = std::vector<int64_t>{1, 0, 1, 1};
        celeg::CheckpointView flagged_checkpoint;
        flagged_checkpoint.metadata = std::move(flagged);
        flagged_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto flagged_model = catalog.select(flagged_checkpoint.metadata).resolve(flagged_checkpoint);
        CELEG_TEST_CHECK(flagged_model.graph.layers.size() == 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto& attention = std::get<celeg::AttentionSpec>(
                flagged_model.graph.layers[static_cast<size_t>(layer)].mixer);
            if (layer == 1) {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
                    attention.position));
            } else {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::RopePositionSpec>(
                    attention.position));
            }
        }

        auto ragged_flags = postnorm_evidence_metadata("ragged_rope_model");
        ragged_flags.values["rope_layer_flags"] = std::vector<int64_t>{1, 0};
        CELEG_TEST_CHECK(inference_input_fails_with(
            std::move(ragged_flags), celeg::ResolutionFailureKind::ConflictingMetadata));
    }

    /// `no_rope_layer_interval`: the Lizzy fallback -- without usable flags,
    /// every Nth layer (`(index + 1) % interval == 0`) loses RoPE. A
    /// non-positive interval fails loudly instead of disabling nothing.
    {
        auto interval_metadata = postnorm_evidence_metadata("interval_rope_model");
        interval_metadata.values["no_rope_layer_interval"] = int64_t(2);
        celeg::CheckpointView interval_checkpoint;
        interval_checkpoint.metadata = std::move(interval_metadata);
        interval_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto interval_model = catalog.select(interval_checkpoint.metadata).resolve(interval_checkpoint);
        CELEG_TEST_CHECK(interval_model.graph.layers.size() == 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto& attention = std::get<celeg::AttentionSpec>(
                interval_model.graph.layers[static_cast<size_t>(layer)].mixer);
            if (layer == 1 || layer == 3) {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::NoPositionEncodingSpec>(
                    attention.position));
            } else {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::RopePositionSpec>(
                    attention.position));
            }
        }

        auto bad_interval = postnorm_evidence_metadata("bad_interval_model");
        bad_interval.values["no_rope_layer_interval"] = int64_t(0);
        CELEG_TEST_CHECK(inference_input_fails_with(
            std::move(bad_interval), celeg::ResolutionFailureKind::ConflictingMetadata));
    }
}

}
