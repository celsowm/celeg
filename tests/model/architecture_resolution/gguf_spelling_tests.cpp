#include "gguf_spelling_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace celeg::architecture_resolution_test {

void run_gguf_spelling_tests(const celeg::ArchitectureCatalog& catalog) {
    /// GGUF sliding schedule: without `layer_types`, the per-layer pattern
    /// comes from `<arch>.attention.sliding_window_pattern` (1 = sliding).
    /// A present-but-disagreeing `layer_types` fails loudly.
    {
        celeg::CheckpointMetadata gguf_pattern = postnorm_evidence_metadata("gguf_pattern_model");
        gguf_pattern.source_format = celeg::CheckpointSourceFormat::Gguf;
        gguf_pattern.values["general.architecture"] = std::string("testarch");
        gguf_pattern.values.erase("layer_types");
        gguf_pattern.values["testarch.attention.sliding_window_pattern"] =
            std::vector<int64_t>{1, 1, 0, 1};
        celeg::CheckpointView gguf_pattern_checkpoint;
        gguf_pattern_checkpoint.metadata = std::move(gguf_pattern);
        gguf_pattern_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto gguf_pattern_model = catalog.select(gguf_pattern_checkpoint.metadata)
                                            .resolve(gguf_pattern_checkpoint);
        CELEG_TEST_CHECK(gguf_pattern_model.graph.layers.size() == 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto& attention = std::get<celeg::AttentionSpec>(
                gguf_pattern_model.graph.layers[static_cast<size_t>(layer)].mixer);
            if (layer == 2) {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::FullCausalPattern>(
                    attention.pattern));
            } else {
                CELEG_TEST_CHECK(std::holds_alternative<celeg::SlidingWindowPattern>(
                    attention.pattern));
            }
        }

        auto conflicted_pattern = postnorm_evidence_metadata("gguf_pattern_conflict");
        conflicted_pattern.source_format = celeg::CheckpointSourceFormat::Gguf;
        conflicted_pattern.values["general.architecture"] = std::string("testarch");
        conflicted_pattern.values["layer_types"] = std::vector<std::string>{
            "full_attention", "full_attention", "full_attention", "full_attention"};
        conflicted_pattern.values["testarch.attention.sliding_window_pattern"] =
            std::vector<int64_t>{1, 1, 0, 1};
        CELEG_TEST_CHECK(inference_input_fails_with(
            std::move(conflicted_pattern), celeg::ResolutionFailureKind::ConflictingMetadata));
    }

    /// GGUF YaRN spellings: `<arch>.rope.scaling.*` merges with the flat
    /// keys, so a GGUF checkpoint carrying only arch-suffixed YaRN resolves
    /// identically to its flat-spelled twin.
    {
        celeg::CheckpointMetadata gguf_yarn = postnorm_evidence_metadata("gguf_yarn_model");
        gguf_yarn.source_format = celeg::CheckpointSourceFormat::Gguf;
        gguf_yarn.values["general.architecture"] = std::string("testarch");
        gguf_yarn.values.erase("rope_scaling.rope_type");
        gguf_yarn.values.erase("rope_scaling.factor");
        gguf_yarn.values.erase("rope_scaling.original_max_position_embeddings");
        gguf_yarn.values.erase("rope_scaling.attention_factor");
        gguf_yarn.values.erase("rope_scaling.beta_fast");
        gguf_yarn.values.erase("rope_scaling.beta_slow");
        gguf_yarn.values["testarch.rope.scaling.type"] = std::string("yarn");
        gguf_yarn.values["testarch.rope.scaling.factor"] = 8.0;
        gguf_yarn.values["testarch.rope.scaling.original_context_length"] = int64_t(8192);
        gguf_yarn.values["testarch.rope.scaling.yarn_attn_factor"] = 1.2079441541679836;
        gguf_yarn.values["testarch.rope.scaling.yarn_beta_fast"] = 32.0;
        gguf_yarn.values["testarch.rope.scaling.yarn_beta_slow"] = 1.0;
        celeg::CheckpointView gguf_yarn_checkpoint;
        gguf_yarn_checkpoint.metadata = std::move(gguf_yarn);
        gguf_yarn_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto gguf_yarn_model = catalog.select(gguf_yarn_checkpoint.metadata)
                                         .resolve(gguf_yarn_checkpoint);
        const auto& yarn_attention = std::get<celeg::AttentionSpec>(
            gguf_yarn_model.graph.layers[0].mixer);
        const auto& yarn_rope = std::get<celeg::RopePositionSpec>(yarn_attention.position);
        const auto& yarn_scaling = std::get<celeg::YarnRopeScaling>(yarn_rope.scaling);
        CELEG_TEST_CHECK(yarn_scaling.factor == 8.0);
        CELEG_TEST_CHECK(yarn_scaling.original_context == 8192);
    }
}

}
