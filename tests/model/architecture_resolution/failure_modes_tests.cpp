#include "failure_modes_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

#include <string>
#include <vector>

namespace celeg::architecture_resolution_test {

void run_failure_modes_tests(const celeg::ArchitectureCatalog& catalog) {
    auto truncated_norm_schedule = structural_metadata("unknown_norm_schedule");
    truncated_norm_schedule.values["num_hidden_layers"] = int64_t(2);
    truncated_norm_schedule.values["mixer_pre_norm"] = std::vector<int64_t>{1};
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(truncated_norm_schedule),
        celeg::ResolutionFailureKind::IncompleteLayerSchedule));

    auto disabled_present_norm = structural_metadata("unknown_norm_conflict");
    disabled_present_norm.values["mixer_pre_norm"] = false;
    CELEG_TEST_CHECK(resolution_fails_with(
        catalog, std::move(disabled_present_norm), NormFixtureLayout::PreOnly, false,
        celeg::ResolutionFailureKind::ConflictingInferenceFacts));

    auto required_missing_norm = structural_metadata("unknown_norm_missing");
    required_missing_norm.values["mixer_pre_norm"] = true;
    CELEG_TEST_CHECK(resolution_fails_with(
        catalog, std::move(required_missing_norm), NormFixtureLayout::None, false,
        celeg::ResolutionFailureKind::MissingTensorRole));

    auto disabled_qk_norm = structural_metadata("unknown_qk_conflict");
    disabled_qk_norm.values["qk_norm"] = false;
    CELEG_TEST_CHECK(resolution_fails_with(
        catalog, std::move(disabled_qk_norm), NormFixtureLayout::PreOnly, true,
        celeg::ResolutionFailureKind::ConflictingInferenceFacts));

    auto required_qk_norm = structural_metadata("unknown_qk_missing");
    required_qk_norm.values["qk_norm"] = true;
    CELEG_TEST_CHECK(resolution_fails_with(
        catalog, std::move(required_qk_norm), NormFixtureLayout::PreOnly, false,
        celeg::ResolutionFailureKind::MissingTensorRole));

    auto truncated_schedule = structural_metadata("unknown_test_model");
    truncated_schedule.values["num_hidden_layers"] = int64_t(2);
    truncated_schedule.values["intermediate_size"] = std::vector<int64_t>{1728};
    CELEG_TEST_CHECK(normalize_fails_with(
        std::move(truncated_schedule), celeg::ResolutionFailureKind::IncompleteLayerSchedule));

    auto contradictory_qk_norm = structural_metadata("unknown_test_model");
    contradictory_qk_norm.values["qk_norm"] = true;
    contradictory_qk_norm.values["query_key_norm"] = false;
    CELEG_TEST_CHECK(normalize_fails_with(
        std::move(contradictory_qk_norm), celeg::ResolutionFailureKind::ConflictingMetadata));

    auto unknown_semantic_metadata = structural_metadata("unknown_test_model");
    unknown_semantic_metadata.values["qk_norm_strategy"] = std::string("mystery");
    CELEG_TEST_CHECK(normalize_fails_with(
        std::move(unknown_semantic_metadata),
        celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
}

void run_late_failure_modes_tests(const celeg::ArchitectureCatalog& catalog) {
    auto conflicting_layout = structural_metadata("unknown_layout_conflict");
    conflicting_layout.values["layer_layouts"] =
        std::vector<std::string>{"decoder_postnorm"};
    conflicting_layout.values["use_pre_attn_norm"] = true;
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(conflicting_layout),
        celeg::ResolutionFailureKind::ConflictingMetadata));

    auto incomplete_yarn = structural_metadata("unknown_incomplete_yarn");
    incomplete_yarn.values["rope_scaling.rope_type"] = std::string("yarn");
    incomplete_yarn.values["rope_scaling.factor"] = 4.0;
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(incomplete_yarn),
        celeg::ResolutionFailureKind::MissingRequiredMetadata));

    auto unsupported_scaling = structural_metadata("unknown_scaling_kind");
    unsupported_scaling.values["rope_scaling.rope_type"] = std::string("dynamic");
    unsupported_scaling.values["rope_scaling.factor"] = 2.0;
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(unsupported_scaling),
        celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

    auto full_metadata = structural_metadata("another_unknown_name");
    full_metadata.values["layer_types"] =
        std::vector<std::string>{"full_attention"};
    celeg::CheckpointView full_checkpoint;
    full_checkpoint.metadata = std::move(full_metadata);
    full_checkpoint.repository = std::make_shared<GptxRepository>();
    const auto full_model = catalog.select(full_checkpoint.metadata).resolve(full_checkpoint);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::FullCausalPattern>(
        std::get<celeg::AttentionSpec>(full_model.graph.layers[0].mixer).pattern));

    auto unknown_pattern = structural_metadata("unknown_test_model");
    unknown_pattern.values["layer_types"] =
        std::vector<std::string>{"mystery_attention"};
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(unknown_pattern), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

    auto sliding_without_window = structural_metadata("unknown_test_model");
    sliding_without_window.values["layer_types"] =
        std::vector<std::string>{"sliding_attention"};
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(sliding_without_window), celeg::ResolutionFailureKind::MissingRequiredMetadata));
}

}
