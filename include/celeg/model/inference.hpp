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

// Deterministically combines proposals for one fact. Equal-value proposals
// are merged; equally valid contradictory proposals fail closed.
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
    std::optional<int> intermediate_size;
    std::optional<int> layer_count;
    std::optional<int> query_heads;
    std::optional<int> key_value_heads;
    std::optional<int> head_dim;
    std::optional<int> vocab_size;
    std::optional<int> context_length;
    std::optional<float> norm_epsilon;
    std::optional<double> rope_theta;
    std::optional<float> rotary_fraction;
    std::optional<int> bos_token_id;
    std::vector<int> eos_token_ids;
    std::optional<int> pad_token_id;
    std::optional<bool> query_key_norm;
    std::optional<bool> xsa_projection;
    std::optional<float> xsa_minimum_norm_squared;
    std::optional<bool> tied_embeddings;
    std::optional<RopePairingKind> rope_pairing;
    std::vector<EvidenceItem> evidence;
};

struct TensorInventoryEntry {
    std::string name;
    std::vector<std::int64_t> shape;
    TensorDType dtype = TensorDType::Unknown;
};

// Immutable after construction. It indexes only checkpoint facts; semantic
// TensorRole assignment belongs to the binding solver.
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
};

// Canonical facts are the only input accepted by automatic semantic
// synthesis. They contain no backend types, report objects, or raw metadata
// aliases. Tensor spelling is retained only at this checkpoint boundary so
// the weight plan can address the source repository.
struct CanonicalModelFacts {
    RuntimeTopology topology;
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

class WeightPlanSynthesizer {
public:
    WeightPlan synthesize(const CanonicalModelFacts& facts) const;
};

class ResolutionAssembler {
public:
    ResolvedModel assemble(const CanonicalModelFacts& facts) const;
};

} // namespace celeg
