#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/architecture.hpp"
#include "celeg/model/inference.hpp"
#include "celeg/model/resolved.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/// Shared fixtures for the architecture-resolution test modules: metadata
/// builders, shape-only weight repositories, resolution helpers, and
/// expectation utilities. Modules call the runners from a thin main().
namespace celeg::architecture_resolution_test {

enum class NormFixtureLayout {
    PreOnly,
    PostOnly,
    Sandwich,
    None,
};

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

celeg::CheckpointMetadata structural_metadata(std::string model_type);

celeg::CheckpointMetadata postnorm_evidence_metadata(std::string model_type);

celeg::CheckpointMetadata hybrid_kda_mla_metadata(std::string model_type);

celeg::ResolvedModel resolve_postnorm_evidence(
    const celeg::ArchitectureCatalog& catalog,
    std::string model_type);

celeg::ResolvedModel resolve_hybrid_kda_mla(
    const celeg::ArchitectureCatalog& catalog,
    celeg::CheckpointMetadata metadata);

bool hybrid_resolve_fails_with(celeg::CheckpointMetadata metadata,
                               celeg::ResolutionFailureKind expected);

bool equivalent_weight_plan(const celeg::WeightPlan& a, const celeg::WeightPlan& b);

bool has_weight_role(const celeg::WeightPlan& plan, celeg::TensorRole role, int layer);

celeg::ResolvedModel resolve_structural_identity(const celeg::ArchitectureCatalog& catalog,
                                                 std::string model_type);

celeg::ResolvedModel resolve_norm_layout(const celeg::ArchitectureCatalog& catalog,
                                         NormFixtureLayout norms);

bool resolution_fails_with(const celeg::ArchitectureCatalog& catalog,
                           celeg::CheckpointMetadata metadata,
                           NormFixtureLayout norms,
                           bool query_key_norms,
                           celeg::ResolutionFailureKind expected);

bool normalize_fails_with(celeg::CheckpointMetadata metadata,
                          celeg::ResolutionFailureKind expected);

bool inference_input_fails_with(celeg::CheckpointMetadata metadata,
                                celeg::ResolutionFailureKind expected);

}
