#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "celeg/model/program.hpp"
#include "celeg/runtime/context.hpp"
#include "model/inference/support.hpp"
#include "support/assertions.hpp"

#include <cstdlib>
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

class PostnormEvidenceRepository final : public celeg::IWeightRepository {
public:
    PostnormEvidenceRepository() {
        shapes_["model.embed_tokens.weight"] = {32, 8};
        shapes_["model.norm.weight"] = {8};
        shapes_["lm_head.weight"] = {32, 8};
        for (int layer = 0; layer < 4; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer);
            shapes_[prefix + ".self_attn.q_proj.weight"] = {8, 8};
            shapes_[prefix + ".self_attn.k_proj.weight"] = {8, 8};
            shapes_[prefix + ".self_attn.v_proj.weight"] = {8, 8};
            shapes_[prefix + ".self_attn.o_proj.weight"] = {8, 8};
            shapes_[prefix + ".self_attn.q_norm.weight"] = {4};
            shapes_[prefix + ".self_attn.k_norm.weight"] = {4};
            shapes_[prefix + ".post_attn_norm.weight"] = {8};
            shapes_[prefix + ".mlp.gate_proj.weight"] = {16, 8};
            shapes_[prefix + ".mlp.up_proj.weight"] = {16, 8};
            shapes_[prefix + ".mlp.down_proj.weight"] = {8, 16};
            shapes_[prefix + ".post_mlp_norm.weight"] = {8};
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

celeg::CheckpointMetadata postnorm_evidence_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(8);
    metadata.values["intermediate_size"] = int64_t(16);
    metadata.values["num_hidden_layers"] = int64_t(4);
    metadata.values["num_attention_heads"] = int64_t(2);
    metadata.values["num_key_value_heads"] = int64_t(2);
    metadata.values["head_dim"] = int64_t(4);
    metadata.values["vocab_size"] = int64_t(32);
    metadata.values["max_position_embeddings"] = int64_t(32768);
    metadata.values["bos_token_id"] = int64_t(1);
    metadata.values["eos_token_id"] = int64_t(2);
    metadata.values["pad_token_id"] = int64_t(0);
    metadata.values["norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 500000.0;
    metadata.values["tie_word_embeddings"] = false;
    metadata.values["use_qk_norm"] = true;
    metadata.values["use_pre_attn_norm"] = false;
    metadata.values["use_post_attn_norm"] = true;
    metadata.values["use_pre_mlp_norm"] = false;
    metadata.values["use_post_mlp_norm"] = true;
    metadata.values["layer_types"] = std::vector<std::string>{
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"};
    metadata.values["layer_layouts"] = std::vector<std::string>{
        "decoder_postnorm", "decoder_postnorm", "decoder_postnorm", "decoder_postnorm"};
    metadata.values["sliding_window"] = int64_t(4096);
    metadata.values["rope_scaling.rope_type"] = std::string("yarn");
    metadata.values["rope_scaling.factor"] = 8.0;
    metadata.values["rope_scaling.original_max_position_embeddings"] = int64_t(8192);
    metadata.values["rope_scaling.attention_factor"] = 1.2079441541679836;
    metadata.values["rope_scaling.beta_fast"] = 32.0;
    metadata.values["rope_scaling.beta_slow"] = 1.0;
    return metadata;
}

celeg::ResolvedModel resolve_postnorm_evidence(
    const celeg::ArchitectureCatalog& catalog,
    std::string model_type) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = postnorm_evidence_metadata(std::move(model_type));
    checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

/// Four-layer hybrid fixture mirroring Ling's grammar: KDA layers (separate
/// q/k/v/f/b/g projections with per-stream convolutions) everywhere except
/// the last layer of each group, which is factorized latent attention with a
/// head-wise output gate. No feed-forward tensors, so every layer resolves
/// with a monostate feed-forward and the mixer assertions stand alone.
class HybridKdaMlaRepository final : public celeg::IWeightRepository {
public:
    HybridKdaMlaRepository() {
        shapes_["model.embed_tokens.weight"] = {32, 8};
        shapes_["model.norm.weight"] = {8};
        shapes_["lm_head.weight"] = {32, 8};
        for (int layer = 0; layer < 4; ++layer) {
            const std::string prefix = "model.layers." + std::to_string(layer);
            shapes_[prefix + ".input_layernorm.weight"] = {8};
            if (layer == 3) {
                shapes_[prefix + ".attention.q_a_proj.weight"] = {4, 8};
                shapes_[prefix + ".attention.q_a_layernorm.weight"] = {4};
                shapes_[prefix + ".attention.q_b_proj.weight"] = {12, 4};
                shapes_[prefix + ".attention.kv_a_proj_with_mqa.weight"] = {4, 8};
                shapes_[prefix + ".attention.kv_a_layernorm.weight"] = {2};
                shapes_[prefix + ".attention.kv_b_proj.weight"] = {16, 2};
                shapes_[prefix + ".attention.dense.weight"] = {8, 8};
                shapes_[prefix + ".attention.g_proj.weight"] = {2, 8};
            } else {
                for (const std::string& proj :
                     {"q_proj.weight", "k_proj.weight", "v_proj.weight",
                      "f_proj.weight", "g_proj.weight"}) {
                    shapes_[prefix + ".attention." + proj] = {8, 8};
                }
                shapes_[prefix + ".attention.b_proj.weight"] = {2, 8};
                for (const std::string& conv :
                     {"q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"}) {
                    shapes_[prefix + ".attention." + conv] = {8, 1, 4};
                }
                shapes_[prefix + ".attention.dt_bias"] = {8};
                shapes_[prefix + ".attention.A_log"] = {2};
                shapes_[prefix + ".attention.o_norm.weight"] = {4};
                shapes_[prefix + ".attention.o_proj.weight"] = {8, 8};
            }
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

celeg::CheckpointMetadata hybrid_kda_mla_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(8);
    metadata.values["num_hidden_layers"] = int64_t(4);
    metadata.values["num_attention_heads"] = int64_t(2);
    metadata.values["head_dim"] = int64_t(4);
    metadata.values["v_head_dim"] = int64_t(4);
    metadata.values["q_lora_rank"] = int64_t(4);
    metadata.values["kv_lora_rank"] = int64_t(2);
    metadata.values["qk_nope_head_dim"] = int64_t(4);
    metadata.values["qk_rope_head_dim"] = int64_t(2);
    metadata.values["vocab_size"] = int64_t(32);
    metadata.values["max_position_embeddings"] = int64_t(128);
    metadata.values["bos_token_id"] = int64_t(1);
    metadata.values["eos_token_id"] = int64_t(2);
    metadata.values["pad_token_id"] = int64_t(0);
    metadata.values["norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 100000.0;
    metadata.values["tie_word_embeddings"] = false;
    metadata.values["short_conv_kernel_size"] = int64_t(4);
    metadata.values["kda_safe_gate"] = true;
    metadata.values["kda_lower_bound"] = -5.0;
    /// Ling's exact hybrid key set: schedule + projection style + gate
    /// granularity are consumed and verified, the rest are proven inert.
    metadata.values["layer_group_size"] = int64_t(4);
    metadata.values["no_kda_lora"] = true;
    metadata.values["gated_attention_proj_granularity_type"] = std::string("head_wise");
    metadata.values["linear_silu"] = true;
    metadata.values["group_norm_size"] = int64_t(1);
    metadata.values["num_kv_heads_for_linear_attn"] = int64_t(0);
    metadata.values["embedding_dropout"] = 0.0;
    metadata.values["use_mla_nope"] = false;
    return metadata;
}

celeg::ResolvedModel resolve_hybrid_kda_mla(
    const celeg::ArchitectureCatalog& catalog,
    celeg::CheckpointMetadata metadata) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<HybridKdaMlaRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

bool hybrid_resolve_fails_with(celeg::CheckpointMetadata metadata,
                               celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<HybridKdaMlaRepository>();
    try {
        const auto runtime = celeg::create_builtin_runtime_context();
        (void)runtime->architectures().select(checkpoint.metadata).resolve(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

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

    const auto evidence_a = resolve_postnorm_evidence(catalog, "arbitrary_evidence_model");
    const auto evidence_b = resolve_postnorm_evidence(catalog, "another_arbitrary_identity");
    CELEG_TEST_CHECK(evidence_a.graph.fingerprint() == evidence_b.graph.fingerprint());
    CELEG_TEST_CHECK(equivalent_weight_plan(evidence_a.weight_plan, evidence_b.weight_plan));
    CELEG_TEST_CHECK(evidence_a.graph.layers.size() == 4);
    for (int layer = 0; layer < 4; ++layer) {
        const auto& semantic_layer = evidence_a.graph.layers[static_cast<size_t>(layer)];
        CELEG_TEST_CHECK(!semantic_layer.mixer_norm.before.has_value());
        CELEG_TEST_CHECK(semantic_layer.mixer_norm.after.has_value());
        CELEG_TEST_CHECK(!semantic_layer.feed_forward_norm.before.has_value());
        CELEG_TEST_CHECK(semantic_layer.feed_forward_norm.after.has_value());
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionPostNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::FfnOutputNorm, layer));
        CELEG_TEST_CHECK(!has_weight_role(evidence_a.weight_plan,
                                          celeg::TensorRole::AttentionInputNorm, layer));
        CELEG_TEST_CHECK(!has_weight_role(evidence_a.weight_plan,
                                          celeg::TensorRole::FfnInputNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionQueryNorm, layer));
        CELEG_TEST_CHECK(has_weight_role(evidence_a.weight_plan,
                                         celeg::TensorRole::AttentionKeyNorm, layer));
        const auto& attention = std::get<celeg::AttentionSpec>(semantic_layer.mixer);
        CELEG_TEST_CHECK(attention.query_norm.has_value());
        CELEG_TEST_CHECK(attention.key_norm.has_value());
        if (layer < 3) {
            CELEG_TEST_CHECK(std::holds_alternative<celeg::SlidingWindowPattern>(
                attention.pattern));
            CELEG_TEST_CHECK(std::get<celeg::SlidingWindowPattern>(attention.pattern).window ==
                             4096);
        } else {
            CELEG_TEST_CHECK(std::holds_alternative<celeg::FullCausalPattern>(
                attention.pattern));
        }
        const auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
        const auto& scaling = std::get<celeg::YarnRopeScaling>(rope.scaling);
        CELEG_TEST_CHECK(scaling.factor == 8.0);
        CELEG_TEST_CHECK(scaling.original_context == 8192);
    }

    /// Per-pattern RoPE: nested `rope_parameters.<layer_type>.*` fans out over
    /// the `layer_types` schedule so sliding and full layers carry distinct
    /// thetas (and rotary fractions) instead of a first-wins global. No test
    /// fed these nested keys before; without the fan-out the sliding layers
    /// would silently inherit the full theta.
    {
        celeg::CheckpointMetadata nested = postnorm_evidence_metadata("nested_rope_model");
        nested.values["rope_parameters.sliding_attention.rope_theta"] = int64_t(10000);
        nested.values["rope_parameters.full_attention.rope_theta"] = int64_t(1000000);
        nested.values["rope_parameters.full_attention.partial_rotary_factor"] = 0.5;
        celeg::CheckpointView nested_checkpoint;
        nested_checkpoint.metadata = std::move(nested);
        nested_checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
        const auto nested_model = catalog.select(nested_checkpoint.metadata).resolve(nested_checkpoint);
        CELEG_TEST_CHECK(nested_model.graph.layers.size() == 4);
        for (int layer = 0; layer < 4; ++layer) {
            const auto& semantic_layer = nested_model.graph.layers[static_cast<size_t>(layer)];
            const auto& attention = std::get<celeg::AttentionSpec>(semantic_layer.mixer);
            const auto& rope = std::get<celeg::RopePositionSpec>(attention.position);
            if (layer < 3) {
                CELEG_TEST_CHECK(rope.theta == 10000.0);
                CELEG_TEST_CHECK(rope.rotary_fraction == 1.0);
            } else {
                CELEG_TEST_CHECK(rope.theta == 1000000.0);
                CELEG_TEST_CHECK(std::abs(rope.rotary_fraction - 0.5) < 1.0e-6);
            }
            /// Global YaRN still applies (no nested `rope_type` overrides it).
            CELEG_TEST_CHECK(std::holds_alternative<celeg::YarnRopeScaling>(rope.scaling));
        }
    }

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

    /// Stated-and-verified keys: `position_embedding_type`, `mlp_type` and
    /// `norm_type` confirm the resolved default and fail loudly on anything
    /// else, so a checkpoint asking for a different policy never resolves
    /// as RoPE/SwiGLU/RMS silently.
    {
        auto policy = structural_metadata("unknown_policy_rope");
        policy.values["position_embedding_type"] = std::string("rope");
        celeg::CheckpointView policy_checkpoint;
        policy_checkpoint.metadata = std::move(policy);
        policy_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(policy_checkpoint.metadata).resolve(policy_checkpoint);

        auto absolute = structural_metadata("unknown_policy_absolute");
        absolute.values["position_embedding_type"] = std::string("absolute");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(absolute), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

        auto gated = structural_metadata("unknown_mlp_gated");
        gated.values["mlp_type"] = std::string("gated");
        celeg::CheckpointView gated_checkpoint;
        gated_checkpoint.metadata = std::move(gated);
        gated_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(gated_checkpoint.metadata).resolve(gated_checkpoint);

        auto linear_mlp = structural_metadata("unknown_mlp_linear");
        linear_mlp.values["mlp_type"] = std::string("linear");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(linear_mlp), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));

        auto rms = structural_metadata("unknown_norm_rms");
        rms.values["norm_type"] = std::string("rmsnorm");
        celeg::CheckpointView rms_checkpoint;
        rms_checkpoint.metadata = std::move(rms);
        rms_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(rms_checkpoint.metadata).resolve(rms_checkpoint);

        auto layer_norm = structural_metadata("unknown_norm_layer");
        layer_norm.values["norm_type"] = std::string("layernorm");
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(layer_norm), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
    }

    /// A nested `rope_scaling.rope_theta` restates the global theta and must
    /// agree with it instead of silently winning.
    {
        auto restated = structural_metadata("unknown_theta_restated");
        restated.values["rope_scaling.rope_theta"] = 100000.0;
        celeg::CheckpointView restated_checkpoint;
        restated_checkpoint.metadata = std::move(restated);
        restated_checkpoint.repository = std::make_shared<GptxRepository>();
        (void)catalog.select(restated_checkpoint.metadata).resolve(restated_checkpoint);

        auto conflicted = structural_metadata("unknown_theta_conflict");
        conflicted.values["rope_scaling.rope_theta"] = 999.0;
        CELEG_TEST_CHECK(normalize_fails_with(
            std::move(conflicted), celeg::ResolutionFailureKind::ConflictingMetadata));
    }
    /// Falsy opt-in flags are provably inert: only a set flag may fail the
    /// gate, so `use_qkv_bias: false` never blocks resolution while
    /// `use_qkv_bias: true` without resolver support still fails loudly.
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{false}));
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{int64_t(0)}));
    CELEG_TEST_CHECK(celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{0.0}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{true}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{int64_t(1)}));
    CELEG_TEST_CHECK(!celeg::inference_detail::metadata_value_is_falsy(
        celeg::MetadataValue{std::string("false")}));

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

    /// Ling-style hybrid: KDA layers resolve to the factorized gated-delta
    /// spec, the group-schedule layer resolves to latent attention with a
    /// head-wise gate, and the whole hybrid key set is ledger-clean under
    /// strict semantics. Strict is scoped to this block so the remaining
    /// tests keep their default warn-mode standing.
    {
        const char* previous_strict = std::getenv("CELEG_STRICT_SEMANTICS");
        const bool had_strict = previous_strict != nullptr;
        const std::string saved_strict = had_strict ? std::string(previous_strict) : std::string();
        setenv("CELEG_STRICT_SEMANTICS", "1", 1);
        const auto hybrid = resolve_hybrid_kda_mla(catalog, hybrid_kda_mla_metadata("ling_hybrid"));
        CELEG_TEST_CHECK(hybrid.graph.layers.size() == 4);
        for (int layer = 0; layer < 3; ++layer) {
            const auto& mixer = std::get<celeg::GatedDeltaNetSpec>(
                hybrid.graph.layers[static_cast<size_t>(layer)].mixer);
            CELEG_TEST_CHECK(mixer.vector_decay);
            CELEG_TEST_CHECK(mixer.safe_decay);
            CELEG_TEST_CHECK(mixer.factorized_projections);
            CELEG_TEST_CHECK(mixer.sigmoid_output_gate);
        }
        const auto& mla = std::get<celeg::AttentionSpec>(hybrid.graph.layers[3].mixer);
        CELEG_TEST_CHECK(mla.output_gate.has_value());
        CELEG_TEST_CHECK(mla.output_gate->granularity ==
                         celeg::AttentionGateGranularity::HeadWise);
        CELEG_TEST_CHECK(mla.output_gate_width() == 2);

        /// A stated gate granularity that disagrees with the `g_proj` shape
        /// is a configuration/weight mismatch, not a resolvable model.
        {
            auto mismatch = hybrid_kda_mla_metadata("ling_gate_mismatch");
            mismatch.values["gated_attention_proj_granularity_type"] =
                std::string("element_wise");
            CELEG_TEST_CHECK(hybrid_resolve_fails_with(
                std::move(mismatch), celeg::ResolutionFailureKind::ConflictingMetadata));
        }

        /// `no_kda_lora: false` selects the LoRA-factorized KDA dialect no
        /// rule binds; it must fail loudly rather than misresolve.
        {
            auto lora = hybrid_kda_mla_metadata("ling_kda_lora");
            lora.values["no_kda_lora"] = false;
            CELEG_TEST_CHECK(hybrid_resolve_fails_with(
                std::move(lora), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
        }

        /// A schedule claim the grammar-resolved mixers contradict (group 2
        /// would make layers 1 and 3 attention) fails loudly.
        {
            auto schedule = hybrid_kda_mla_metadata("ling_schedule_mismatch");
            schedule.values["layer_group_size"] = int64_t(2);
            CELEG_TEST_CHECK(hybrid_resolve_fails_with(
                std::move(schedule), celeg::ResolutionFailureKind::ConflictingMetadata));
        }

        /// Degenerate proven-inert values are ledger-clean; non-degenerate
        /// ones stay loud.
        {
            auto grouped = hybrid_kda_mla_metadata("ling_grouped_norm");
            grouped.values["group_norm_size"] = int64_t(2);
            CELEG_TEST_CHECK(hybrid_resolve_fails_with(
                std::move(grouped), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
            auto kv_heads = hybrid_kda_mla_metadata("ling_linear_kv_heads");
            kv_heads.values["num_kv_heads_for_linear_attn"] = int64_t(2);
            CELEG_TEST_CHECK(hybrid_resolve_fails_with(
                std::move(kv_heads), celeg::ResolutionFailureKind::UnsupportedSemanticFeature));
        }
        if (had_strict) {
            setenv("CELEG_STRICT_SEMANTICS", saved_strict.c_str(), 1);
        } else {
            unsetenv("CELEG_STRICT_SEMANTICS");
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
