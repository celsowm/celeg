#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "celeg/model/program.hpp"
#include "celeg/runtime/context.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class TestArchitecture final : public celeg::IArchitecture {
public:
    TestArchitecture(std::string name, int specificity)
        : name_(std::move(name)), specificity_(specificity) {}

    std::string_view id() const override { return name_; }
    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {true, specificity_, "test"};
    }
    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override { return {}; }

private:
    std::string name_;
    int specificity_;
};

celeg::CheckpointMetadata structural_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(576);
    metadata.values["intermediate_size"] = int64_t(1728);
    metadata.values["num_hidden_layers"] = int64_t(1);
    metadata.values["num_attention_heads"] = int64_t(9);
    metadata.values["num_key_value_heads"] = int64_t(3);
    metadata.values["head_dim"] = int64_t(64);
    metadata.values["vocab_size"] = int64_t(32770);
    metadata.values["max_position_embeddings"] = int64_t(8192);
    metadata.values["bos_token_id"] = int64_t(0);
    metadata.values["eos_token_id"] = int64_t(0);
    metadata.values["pad_token_id"] = int64_t(1);
    metadata.values["rms_norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 100000.0;
    metadata.values["tie_word_embeddings"] = true;
    return metadata;
}

enum class NormFixtureLayout {
    PreOnly,
    PostOnly,
    Sandwich,
    None,
};

class GptxRepository final : public celeg::IWeightRepository {
public:
    explicit GptxRepository(NormFixtureLayout norms = NormFixtureLayout::PreOnly,
                            bool query_key_norms = false) {
        shapes_["transformer.wte.weight"] = {32770, 576};
        shapes_["transformer.ln_f.weight"] = {576};
        for (int layer = 0; layer < 1; ++layer) {
            const std::string prefix = "transformer.h." + std::to_string(layer);
            const std::string model_prefix =
                "model.layers." + std::to_string(layer);
            if (norms == NormFixtureLayout::PreOnly ||
                norms == NormFixtureLayout::Sandwich) {
                shapes_[prefix + ".ln_1.weight"] = {576};
                shapes_[prefix + ".ln_2.weight"] = {576};
            }
            if (norms == NormFixtureLayout::PostOnly ||
                norms == NormFixtureLayout::Sandwich) {
                shapes_[model_prefix + ".post_attention_layernorm.weight"] = {576};
                shapes_[model_prefix + ".post_feedforward_layernorm.weight"] = {576};
            }
            shapes_[prefix + ".attn.q_proj.weight"] = {576, 576};
            shapes_[prefix + ".attn.k_proj.weight"] = {192, 576};
            shapes_[prefix + ".attn.v_proj.weight"] = {192, 576};
            shapes_[prefix + ".attn.o_proj.weight"] = {576, 576};
            if (query_key_norms) {
                shapes_["blk." + std::to_string(layer) + ".attn_q_norm.weight"] = {64};
                shapes_["blk." + std::to_string(layer) + ".attn_k_norm.weight"] = {64};
            }
            shapes_[prefix + ".mlp.w_gate.weight"] = {1728, 576};
            shapes_[prefix + ".mlp.w_up.weight"] = {1728, 576};
            shapes_[prefix + ".mlp.w_down.weight"] = {576, 1728};
        }
    }

    bool contains(std::string_view name) const override {
        return shapes_.contains(std::string(name));
    }

    celeg::HostTensorView tensor(std::string_view name) const override {
        return {celeg::TensorDType::BF16, shapes_.at(std::string(name)), nullptr, 0};
    }

    std::vector<std::string> names() const override {
        std::vector<std::string> result;
        result.reserve(shapes_.size());
        for (const auto& [name, shape] : shapes_) {
            (void)shape;
            result.push_back(name);
        }
        return result;
    }

private:
    std::unordered_map<std::string, std::vector<int64_t>> shapes_;
};

bool equivalent_weight_plan(const celeg::WeightPlan& a, const celeg::WeightPlan& b) {
    if (a.requests.size() != b.requests.size()) return false;
    for (std::size_t index = 0; index < a.requests.size(); ++index) {
        const auto& lhs = a.requests[index];
        const auto& rhs = b.requests[index];
        if (lhs.role != rhs.role || lhs.layer != rhs.layer || lhs.expert != rhs.expert ||
            lhs.expected_shape != rhs.expected_shape || lhs.source_name != rhs.source_name ||
            lhs.physical_layer != rhs.physical_layer ||
            lhs.norm_weight_kind != rhs.norm_weight_kind) {
            return false;
        }
    }
    return true;
}

bool has_weight_role(const celeg::WeightPlan& plan, celeg::TensorRole role, int layer) {
    for (const auto& request : plan.requests) {
        if (request.role == role && request.layer == layer) return true;
    }
    return false;
}

celeg::ResolvedModel resolve_structural_identity(const celeg::ArchitectureCatalog& catalog,
                                                 std::string model_type) {
    auto metadata = structural_metadata(std::move(model_type));
    metadata.repository_hint = "synthetic/identity-invariance";
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

celeg::ResolvedModel resolve_norm_layout(const celeg::ArchitectureCatalog& catalog,
                                         NormFixtureLayout norms) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = structural_metadata("unknown_norm_layout");
    checkpoint.repository = std::make_shared<GptxRepository>(norms);
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

bool resolution_fails_with(const celeg::ArchitectureCatalog& catalog,
                           celeg::CheckpointMetadata metadata,
                           NormFixtureLayout norms,
                           bool query_key_norms,
                           celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>(norms, query_key_norms);
    try {
        (void)catalog.select(checkpoint.metadata).resolve(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

bool normalize_fails_with(celeg::CheckpointMetadata metadata,
                          celeg::ResolutionFailureKind expected) {
    try {
        (void)celeg::normalize_model_metadata(metadata);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

bool inference_input_fails_with(celeg::CheckpointMetadata metadata,
                                celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>();
    try {
        (void)celeg::build_inference_input(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

}

int main() {
    const auto runtime = celeg::create_builtin_runtime_context();
    const auto& catalog = runtime->architectures();
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

    const auto identity_a = resolve_structural_identity(catalog, "lizzy");
    const auto identity_b = resolve_structural_identity(catalog, "unknown_test_model");
    const auto poisoned_identity = resolve_structural_identity(catalog, "gemma4");
    CELEG_TEST_CHECK(identity_a.graph.fingerprint() == identity_b.graph.fingerprint());
    CELEG_TEST_CHECK(identity_a.graph.fingerprint() == poisoned_identity.graph.fingerprint());
    CELEG_TEST_CHECK(equivalent_weight_plan(identity_a.weight_plan, identity_b.weight_plan));
    CELEG_TEST_CHECK(equivalent_weight_plan(identity_a.weight_plan,
                                            poisoned_identity.weight_plan));

    const auto pre_only = resolve_norm_layout(catalog, NormFixtureLayout::PreOnly);
    CELEG_TEST_CHECK(pre_only.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(!pre_only.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(pre_only.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(!pre_only.graph.layers[0].feed_forward_norm.after.has_value());
    CELEG_TEST_CHECK(has_weight_role(pre_only.weight_plan,
                                     celeg::TensorRole::AttentionInputNorm, 0));
    CELEG_TEST_CHECK(!has_weight_role(pre_only.weight_plan,
                                      celeg::TensorRole::AttentionPostNorm, 0));

    const auto post_only = resolve_norm_layout(catalog, NormFixtureLayout::PostOnly);
    CELEG_TEST_CHECK(!post_only.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(post_only.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(!post_only.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(post_only.graph.layers[0].feed_forward_norm.after.has_value());
    CELEG_TEST_CHECK(!has_weight_role(post_only.weight_plan,
                                      celeg::TensorRole::AttentionInputNorm, 0));
    CELEG_TEST_CHECK(has_weight_role(post_only.weight_plan,
                                     celeg::TensorRole::AttentionPostNorm, 0));
    CELEG_TEST_CHECK(!has_weight_role(post_only.weight_plan,
                                      celeg::TensorRole::FfnInputNorm, 0));
    CELEG_TEST_CHECK(has_weight_role(post_only.weight_plan,
                                     celeg::TensorRole::FfnOutputNorm, 0));

    const auto sandwich = resolve_norm_layout(catalog, NormFixtureLayout::Sandwich);
    CELEG_TEST_CHECK(sandwich.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].feed_forward_norm.after.has_value());

    const auto no_norm = resolve_norm_layout(catalog, NormFixtureLayout::None);
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].feed_forward_norm.after.has_value());

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
    full_metadata.values["layer_layouts"] =
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

    auto truncated_pattern = structural_metadata("unknown_test_model");
    truncated_pattern.values["num_hidden_layers"] = int64_t(2);
    truncated_pattern.values["layer_types"] =
        std::vector<std::string>{"full_attention"};
    CELEG_TEST_CHECK(inference_input_fails_with(
        std::move(truncated_pattern), celeg::ResolutionFailureKind::IncompleteLayerSchedule));

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

    celeg::ArchitectureCatalog mutable_catalog;
    mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    bool duplicate_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    CELEG_TEST_CHECK(duplicate_rejected);
    mutable_catalog.freeze();
    bool mutation_rejected = false;
    try {
        mutable_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    } catch (const std::logic_error&) {
        mutation_rejected = true;
    }
    CELEG_TEST_CHECK(mutation_rejected);

    celeg::ArchitectureCatalog ambiguous_catalog;
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("one", 10));
    ambiguous_catalog.add(std::make_unique<TestArchitecture>("two", 10));
    bool ambiguity_rejected = false;
    try {
        ambiguous_catalog.select(metadata);
    } catch (const std::runtime_error&) {
        ambiguity_rejected = true;
    }
    CELEG_TEST_CHECK(ambiguity_rejected);

    std::cout << "architecture_resolution_test: ok\n";
}
