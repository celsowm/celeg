#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/view.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/resolved.hpp"

#include <algorithm>
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
    RopeScalingSpec scaling;

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

    friend bool operator==(const LayerScopedValue& left,
                           const LayerScopedValue& right) {
        const size_t layers = std::max(left.per_layer.size(), right.per_layer.size());
        if (layers == 0) return left.global == right.global;
        for (size_t layer = 0; layer < layers; ++layer) {
            if (left.value_for(static_cast<int>(layer)) !=
                right.value_for(static_cast<int>(layer))) {
                return false;
            }
        }
        return true;
    }
};

enum class AttentionPatternKind : std::uint8_t {
    /// Layer carries no attention mixer (e.g. a short-convolution or recurrent
    /// mixer resolved by its own inference rule); the attention-pattern schedule
    /// is simply not applicable to it.
    None,
    FullCausal,
    SlidingWindow,
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

/// Decay parameter encoding in checkpoint weights: LogA (log-space parameter,
/// needs exp transform at load/runtime) or Pretransformed (linear/already-decayed).
enum class DecayParameterEncoding : std::uint8_t {
    LogA,
    Pretransformed,
};

/// Scoring function for MoE routing.
enum class MoeRouterScoreFunction : std::uint8_t {
    Sigmoid,
    Softmax,
};

/// Expert selection algorithm for MoE routing.
enum class MoeRouterSelectionMethod : std::uint8_t {
    TopK,
    Greedy,
    NoauxTc,
};

/// Core architectural dimensions and global parameters shared across layer families.
struct CoreModelFacts {
    std::optional<int> hidden_size;
    LayerScopedValue<int> intermediate_size;
    std::optional<int> layer_count;
    std::optional<int> vocab_size;
    std::optional<int> context_length;
    std::optional<float> norm_epsilon;
    std::optional<float> embedding_multiplier;
    std::optional<float> residual_multiplier;
    std::optional<float> logits_multiplier;
    std::optional<float> logits_divisor;
    std::optional<int> bos_token_id;
    std::vector<int> eos_token_ids;
    std::optional<int> pad_token_id;
    std::optional<bool> tied_embeddings;
    std::optional<bool> feed_forward_auto_adjust;
};

/// Facts governing attention layer geometry, normalization, and positional encoding.
struct AttentionFacts {
    LayerScopedValue<int> query_heads;
    LayerScopedValue<int> key_value_heads;
    LayerScopedValue<int> head_dim;
    LayerScopedValue<AttentionPatternKind> pattern;
    std::optional<int> sliding_window;
    std::optional<float> attention_multiplier;
    InferredPositionEncoding position_encoding = UnresolvedPositionEncoding{};
    std::optional<bool> query_key_norm;
    std::optional<bool> xsa_projection;
    std::optional<float> xsa_minimum_norm_squared;
};

/// Facts governing before/after normalization placement around each sublayer.
struct StructuralNormFacts {
    LayerScopedValue<bool> mixer_before;
    LayerScopedValue<bool> mixer_after;
    LayerScopedValue<bool> feed_forward_before;
    LayerScopedValue<bool> feed_forward_after;
};

/// Facts governing factorized/multi-head latent attention (MLA).
struct LatentAttentionFacts {
    std::optional<int> query_rank;
    std::optional<int> kv_rank;
    std::optional<int> query_head_dim;
    std::optional<int> query_nope_dim;
    std::optional<int> query_rope_dim;
    std::optional<int> value_head_dim;
};

/// Facts governing short-convolution mixer layers.
struct ShortConvolutionFacts {
    std::optional<int> cache_length;
};

/// Facts governing recurrent gated-delta-net / linear-attention layers.
struct GatedDeltaFacts {
    std::optional<int> conv_kernel;
    std::optional<int> key_heads;
    std::optional<int> value_heads;
    std::optional<int> key_dim;
    std::optional<int> value_dim;
    std::optional<bool> safe_decay;
    std::optional<float> decay_lower_bound;
    DecayParameterEncoding decay_encoding = DecayParameterEncoding::LogA;
};

/// Facts governing Mamba-2 SSM mixer layers.
struct Mamba2Facts {
    std::optional<int> intermediate;
    std::optional<int> state_size;
    std::optional<int> time_step_rank;
    std::optional<int> num_heads;
    std::optional<int> head_dim;
    std::optional<int> group_count;
    std::optional<int> conv_kernel;
    std::optional<int> chunk_size;
    DecayParameterEncoding decay_encoding = DecayParameterEncoding::LogA;
};

/// Facts governing mixture-of-experts (MoE) routing and layout.
struct MoeFacts {
    std::optional<int> first_dense_layer;
    std::optional<int> experts;
    std::optional<int> experts_per_token;
    std::optional<int> intermediate;
    std::optional<int> shared_intermediate;
    std::optional<int> routing_groups;
    std::optional<int> total_routing_groups;
    std::optional<int> group_score_top_k;
    std::optional<bool> normalize_topk;
    std::optional<bool> expert_bias;
    std::optional<float> routed_scaling;
    std::optional<MoeRouterScoreFunction> score_function;
    std::optional<MoeRouterSelectionMethod> selection_method;
};

/// Normalized checkpoint metadata decomposed into typed fact groups.
struct NormalizedModelMetadata {
    CoreModelFacts core;
    AttentionFacts attention;
    StructuralNormFacts norms;
    LatentAttentionFacts latent_attention;
    ShortConvolutionFacts short_conv;
    GatedDeltaFacts gated_delta;
    Mamba2Facts mamba2;
    MoeFacts moe;
    std::vector<EvidenceItem> evidence;
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
    std::string architecture_type;

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
    // `repository` lets weight-plan resolution confirm candidate tensor
    // names actually exist in the checkpoint rather than trusting the
    // first candidate blindly. Pass nullptr only when no checkpoint
    // repository is available (e.g. synthetic facts in tests).
    ResolvedModel assemble(const CanonicalModelFacts& facts,
                           const IWeightRepository* repository) const;
};

}
