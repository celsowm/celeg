#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

enum class CompiledMixer : unsigned char { Attention, ShortConvolution };
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts };

// Neutral, immutable MoE semantics.  These values are intentionally free of
// checkpoint names, architecture identity, CUDA handles, and cache policy.
enum class MoeRouterScoreKind : unsigned char { SigmoidProbabilities, SoftmaxLogits };
enum class MoeSelectionKind : unsigned char { TopK, GroupedTopK };
enum class MoeNormalizationKind : unsigned char { None, SumSelected };
enum class MoePayloadLayout : unsigned char { Individual, Stacked, Fused };
enum class MoePayloadDType : unsigned char { Unknown, BF16, F16, F32, I8, Quantized };
enum class MoeActivation : unsigned char { SwiGLU, GeluTanh };
enum class MoeCombineOrder : unsigned char { RoutedThenShared, SharedThenRouted };

struct ExpertPayloadRegion {
    TensorRole role = TensorRole::MoeExpertGate;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    MoePayloadDType dtype = MoePayloadDType::Unknown;
};

struct ExpertPayloadSchema {
    MoePayloadLayout layout = MoePayloadLayout::Individual;
    std::size_t total_bytes = 0;
    std::vector<ExpertPayloadRegion> regions;

    void validate() const;
    std::string fingerprint() const;
};

struct RouterProgram {
    MoeRouterScoreKind score = MoeRouterScoreKind::SigmoidProbabilities;
    MoeSelectionKind selection = MoeSelectionKind::TopK;
    MoeNormalizationKind normalization = MoeNormalizationKind::None;
    int expert_count = 0;
    int experts_per_token = 0;
    int group_count = 0;
    int experts_per_group = 0;
    bool has_expert_bias = false;
    float routed_scaling = 1.0f;

    void validate() const;
    std::string fingerprint() const;
};

struct ExpertMlpProgram {
    MoeActivation activation = MoeActivation::SwiGLU;
    int hidden_size = 0;
    int intermediate_size = 0;

    void validate() const;
    std::string fingerprint() const;
};

struct RoutedExpertProgram {
    ExpertMlpProgram mlp;
    ExpertPayloadSchema payload;
};

struct SharedExpertProgram {
    ExpertMlpProgram mlp;
};

struct MoeOutputProgram {
    bool has_shared_expert = false;
    MoeCombineOrder combine_order = MoeCombineOrder::RoutedThenShared;
};

struct ExpertResidencyRequirements {
    int expert_count = 0;
    std::size_t payload_bytes = 0;
};

struct MoeLayerProgram {
    RouterProgram router;
    RoutedExpertProgram routed;
    std::optional<SharedExpertProgram> shared;
    MoeOutputProgram output;
    ExpertResidencyRequirements residency;

    void validate() const;
    std::string fingerprint() const;
};

struct PerLayerInputPlan {
    bool enabled = false;
    int layer_count = 0;
    int input_size = 0;
    std::size_t packed_width = 0;
    float token_scale = 1.0f;
    float context_scale = 1.0f;
    float residual_scale = 1.0f;
    float norm_epsilon = 0.0f;
    ActivationKind activation = ActivationKind::GeluTanh;

    static PerLayerInputPlan derive(const ResolvedModel& model);
    std::size_t checked_elements(std::size_t rows) const;
    void validate() const;
};

// Immutable execution description produced before a backend starts serving.
// It contains no checkpoint or architecture probing state, so decode can use
// direct indices and function selection instead of format/architecture tests.
struct CompiledLayerProgram {
    CompiledMixer mixer;
    CompiledFeedForward feed_forward;
    std::vector<std::size_t> weight_request_indices;
    std::optional<MoeLayerProgram> moe;
};

struct CompiledModelProgram {
    std::string identity;
    std::size_t weight_request_count = 0;
    std::vector<CompiledLayerProgram> layers;
    std::vector<std::size_t> unlayered_weight_request_indices;
    PerLayerInputPlan per_layer_input;
    std::string semantic_fingerprint;

    bool has_moe() const;
    void validate() const;
};

struct MoeBackendCapabilities {
    bool grouped_selection = false;
    bool shared_experts = false;
    bool stacked_payload = false;
};

// Backend compilers call this after generic semantic compilation and before
// allocating backend state.  Unsupported semantics therefore fail with a
// useful diagnostic instead of reaching a decode loop.
void validate_moe_backend_capabilities(const CompiledModelProgram& program,
                                       std::string_view backend,
                                       MoeBackendCapabilities capabilities);

CompiledModelProgram build_model_program(const ResolvedModel& model);

} // namespace celeg
