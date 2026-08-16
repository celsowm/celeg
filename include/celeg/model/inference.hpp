#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/view.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/resolved.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace celeg {

enum class EvidenceKind : std::uint8_t {
    ExplicitMetadata,
    AliasMetadata,
    TensorShape,
    TensorName,
    FormatGuarantee,
    Derived,
};

struct EvidenceItem {
    EvidenceKind kind = EvidenceKind::Derived;
    std::string source;
    std::string fact;
};

enum class ProposalStrength : std::uint8_t {
    ExplicitMetadata,
    ShapeDerived,
    NamingDerived,
    FormatGuaranteed,
};

template <typename T>
struct InferenceProposal {
    T value;
    std::vector<EvidenceItem> evidence;
    ProposalStrength strength = ProposalStrength::NamingDerived;
    std::string rule_id;

    friend bool operator==(const InferenceProposal&, const InferenceProposal&) = default;
};

// Inferred positional-encoding outcome for a checkpoint's attention layers.
// Exactly one alternative is ever populated, so an inferred RoPE payload can
// never coexist with a "no position encoding" conclusion, and generic
// semantic code consumes only this variant -- never an architecture name.
struct UnresolvedPositionEncoding {};

struct InferredRopePosition {
    double theta = 0.0;
    float rotary_fraction = 1.0f;
    RopePairingKind pairing = RopePairingKind::SplitHalf;

    friend bool operator==(const InferredRopePosition&, const InferredRopePosition&) = default;
};

using InferredPositionEncoding = std::variant<
    UnresolvedPositionEncoding,
    NoPositionEncodingSpec,
    InferredRopePosition>;

template <typename T>
struct LayerScopedValue {
    std::optional<T> global;
    std::vector<std::optional<T>> per_layer;

    bool has_value() const noexcept {
        if (global.has_value()) return true;
        for (const auto& value : per_layer) {
            if (value.has_value()) return true;
        }
        return false;
    }

    std::optional<T> value_for(int layer) const {
        if (layer >= 0 && layer < static_cast<int>(per_layer.size()) &&
            per_layer[static_cast<size_t>(layer)].has_value()) {
            return per_layer[static_cast<size_t>(layer)];
        }
        return global;
    }

    friend bool operator==(const LayerScopedValue&, const LayerScopedValue&) = default;
};

enum class ResolutionFailureKind : std::uint8_t {
    MissingRequiredMetadata,
    ConflictingMetadata,
    ConflictingInferenceFacts,
    AmbiguousTensorBinding,
    MissingTensorRole,
    ShapeConstraintViolation,
    UnsupportedSemanticFeature,
    UnsupportedTensorLayout,
    UnsupportedGraphPrimitive,
    IncompleteLayerSchedule,
    BackendCapabilityMismatch,
};

class ResolutionError final : public std::runtime_error {
public:
    ResolutionError(ResolutionFailureKind kind, std::string message,
                    std::vector<EvidenceItem> evidence = {});

    ResolutionFailureKind kind() const noexcept { return kind_; }
    const std::vector<EvidenceItem>& evidence() const noexcept { return evidence_; }

private:
    ResolutionFailureKind kind_;
    std::vector<EvidenceItem> evidence_;
};

class FactSolver {
public:
    template <typename T>
    InferenceProposal<T> solve(const std::vector<InferenceProposal<T>>& proposals) const {
        if (proposals.empty()) {
            throw ResolutionError(ResolutionFailureKind::MissingRequiredMetadata,
                                 "no proposal was produced for a required fact");
        }
        InferenceProposal<T> result = proposals.front();
        for (size_t index = 1; index < proposals.size(); ++index) {
            if (!(result.value == proposals[index].value)) {
                throw ResolutionError(ResolutionFailureKind::ConflictingInferenceFacts,
                                      "conflicting proposals for one canonical fact");
            }
            result.evidence.insert(result.evidence.end(), proposals[index].evidence.begin(),
                                   proposals[index].evidence.end());
        }
        return result;
    }
};

struct NormalizedModelMetadata {
    std::optional<int> hidden_size;
    LayerScopedValue<int> intermediate_size;
    std::optional<int> layer_count;
    LayerScopedValue<int> query_heads;
    LayerScopedValue<int> key_value_heads;
    LayerScopedValue<int> head_dim;
    std::optional<int> mamba_intermediate;
    std::optional<int> mamba_state_size;
    std::optional<int> mamba_time_step_rank;
    std::optional<int> mamba_num_heads;
    std::optional<int> mamba_head_dim;
    std::optional<int> mamba_group_count;
    std::optional<int> mamba_conv_kernel;
    std::optional<int> mamba_chunk_size;
    std::optional<int> vocab_size;
    std::optional<int> context_length;
    std::optional<float> norm_epsilon;
    std::optional<float> embedding_multiplier;
    std::optional<float> attention_multiplier;
    std::optional<float> residual_multiplier;
    std::optional<float> logits_multiplier;
    std::optional<float> logits_divisor;
    std::optional<int> shortconv_cache;
    InferredPositionEncoding position_encoding = UnresolvedPositionEncoding{};
    std::optional<int> bos_token_id;
    std::vector<int> eos_token_ids;
    std::optional<int> pad_token_id;
    std::optional<bool> query_key_norm;
    std::optional<bool> xsa_projection;
    std::optional<float> xsa_minimum_norm_squared;
    std::optional<bool> tied_embeddings;
    std::vector<EvidenceItem> evidence;
    std::optional<bool> feed_forward_auto_adjust;
    std::optional<int> first_dense_layer;
    std::optional<int> recurrent_conv_kernel;
    std::optional<int> recurrent_key_heads;
    std::optional<int> recurrent_value_heads;
    std::optional<int> recurrent_key_dim;
    std::optional<int> recurrent_value_dim;
    std::optional<bool> recurrent_safe_decay;
    std::optional<float> recurrent_decay_lower_bound;
    std::optional<int> latent_query_rank;
    std::optional<int> latent_kv_rank;
    std::optional<int> latent_query_head_dim;
    std::optional<int> latent_query_nope_dim;
    std::optional<int> latent_query_rope_dim;
    std::optional<int> latent_value_head_dim;
    std::optional<int> moe_experts;
    std::optional<int> moe_experts_per_token;
    std::optional<int> moe_intermediate;
    std::optional<int> moe_shared_intermediate;
    std::optional<int> moe_routing_groups;
    std::optional<int> moe_total_routing_groups;
    std::optional<int> moe_group_score_top_k;
    std::optional<bool> moe_normalize_topk;
    std::optional<bool> moe_expert_bias;
    std::optional<float> moe_routed_scaling;
    std::optional<std::string> moe_score_function;
    std::optional<std::string> moe_selection_method;
};

struct TensorInventoryEntry {
    std::string name;
    std::vector<std::int64_t> shape;
    TensorDType dtype = TensorDType::Unknown;
};

class TensorInventory {
public:
    TensorInventory() = default;
    explicit TensorInventory(std::vector<TensorInventoryEntry> entries);

    const std::vector<TensorInventoryEntry>& entries() const noexcept { return entries_; }
    const TensorInventoryEntry* find(std::string_view name) const noexcept;
    std::vector<const TensorInventoryEntry*> with_prefix(std::string_view prefix) const;

private:
    std::vector<TensorInventoryEntry> entries_;
};

struct TensorRoleBinding {
    TensorRole role;
    int layer = -1;
    int expert = -1;
    int physical_layer = -1;
    std::string source_name;
    std::vector<std::int64_t> shape;
    std::vector<EvidenceItem> evidence;
};

struct TensorRoleBindings {
    std::vector<TensorRoleBinding> values;

    const TensorRoleBinding* find(TensorRole role, int layer = -1,
                                  int expert = -1) const noexcept;
    void validate() const;
};

class BindingSolver {
public:
    TensorRoleBindings solve(const std::vector<TensorRoleBinding>& candidates) const;
};

struct InferenceInput {
    NormalizedModelMetadata metadata;
    TensorInventory inventory;
    CheckpointSourceFormat source_format = CheckpointSourceFormat::Safetensors;

    bool is_gguf() const noexcept { return source_format == CheckpointSourceFormat::Gguf; }
};

struct CanonicalModelFacts {
    CheckpointDimensions checkpoint;
    NumericalPolicy numerical_policy;
    ModelGraph graph;
    TensorRoleBindings bindings;
    bool tied_embeddings = false;
    std::string resolution_mode;
    std::string source_format = "safetensors";
    std::vector<EvidenceItem> evidence;

    std::string fingerprint() const;
    void validate() const;
};

struct ResolutionReport {
    std::string resolution_mode;
    NormalizedModelMetadata metadata;
    std::string chat_template_source;
    std::string chat_program_fingerprint;
    std::string tool_protocol_source;
    std::string vision_pipeline_source;
    std::vector<EvidenceItem> accepted;
    std::vector<EvidenceItem> rejected;
    std::vector<std::string> ambiguities;
    std::vector<std::string> failures;
    std::string canonical_fingerprint;
};

ResolutionReport explain_resolution(const CheckpointView& checkpoint);

NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata);
TensorInventory build_tensor_inventory(const IWeightRepository& repository);
InferenceInput build_inference_input(const CheckpointView& checkpoint);
CanonicalModelFacts infer_canonical_model_facts(const InferenceInput& input);

class GraphSynthesizer {
public:
    ModelGraph synthesize(const CanonicalModelFacts& facts) const;
};

class ResolutionAssembler {
public:
    ResolvedModel assemble(const CanonicalModelFacts& facts) const;
};

}
