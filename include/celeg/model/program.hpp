#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

enum class CompiledMixer : unsigned char {
    Attention, ShortConvolution, GatedDeltaNet, Mamba2, MlpOnly
};
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts };

// Neutral, immutable MoE semantics.  These values are intentionally free of
// checkpoint names, architecture identity, CUDA handles, and cache policy.
enum class MoeRouterScoreKind : unsigned char { SigmoidProbabilities, SoftmaxLogits };
enum class MoeSelectionKind : unsigned char { TopK, GroupedTopK };
enum class MoeNormalizationKind : unsigned char { None, SumSelected };
enum class MoePayloadLayout : unsigned char { Individual, Stacked, Fused };
enum class MoeActivation : unsigned char { SwiGLU, GeluTanh };
enum class MoeCombineOrder : unsigned char { RoutedThenShared, SharedThenRouted };

struct ExpertPayloadRegion {
    TensorRole role = TensorRole::MoeExpertGate;
    std::size_t elements = 0;
};

struct ExpertPayloadSchema {
    MoePayloadLayout layout = MoePayloadLayout::Individual;
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
    std::size_t payload_elements = 0;
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

enum class CompiledStateLayoutKind : uint8_t {
    OrdinaryKv,
    Latent,
};

struct CompiledAttentionStateLayout {
    CompiledStateLayoutKind kind = CompiledStateLayoutKind::OrdinaryKv;
    int key_width = 0;
    int value_width = 0;
    int latent_width = 0;
    int rotary_width = 0;
    std::size_t key_elements = 0;
    std::size_t value_elements = 0;
    std::size_t latent_elements = 0;
    std::size_t rotary_elements = 0;
    std::size_t persistent_elements = 0;
    AttentionStateStorageSpec storage;

    std::size_t scalar_bytes(StateScalarType scalar) const {
        switch (scalar) {
        case StateScalarType::FP32: return sizeof(float);
        case StateScalarType::FP16:
        case StateScalarType::BF16: return sizeof(uint16_t);
        case StateScalarType::FP8:
        case StateScalarType::INT8: return sizeof(uint8_t);
        case StateScalarType::INT4: return 1;
        }
        throw std::invalid_argument("invalid compiled state scalar type");
    }

    std::size_t persistent_bytes() const {
        return key_elements * scalar_bytes(storage.key) +
               value_elements * scalar_bytes(storage.value) +
               latent_elements * scalar_bytes(storage.latent) +
               rotary_elements * scalar_bytes(storage.rotary);
    }

    void validate() const;
};

enum class CompiledChunkCapability : uint8_t {
    Native,
    SequentialAdapter,
    Unsupported,
};

// Immutable execution description produced before a backend starts serving.
// It contains no checkpoint or architecture probing state, so decode can use
// direct indices and function selection instead of format/architecture tests.
struct CompiledLayerProgram {
    CompiledMixer mixer;
    CompiledFeedForward feed_forward;
    bool execute_feed_forward = true;
    CompiledChunkCapability chunk_capability = CompiledChunkCapability::Native;
    std::optional<AttentionSpec> attention;
    std::optional<ShortConvolutionSpec> short_convolution;
    std::optional<GatedDeltaNetSpec> gated_delta_net;
    std::optional<Mamba2Spec> mamba2;
    int feed_forward_intermediate = 0;
    ActivationKind feed_forward_activation = ActivationKind::SwiGLU;
    std::vector<std::size_t> weight_request_indices;
    std::optional<MoeLayerProgram> moe;
    std::optional<CompiledAttentionStateLayout> state_layout;
};

struct CompiledModelProgram {
    std::string identity;
    std::size_t weight_request_count = 0;
    std::vector<CompiledLayerProgram> layers;
    std::vector<int> norm_after_layers;
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
    bool fused_payload = false;
};

// Backend compilers call this after generic semantic compilation and before
// allocating backend state.  Unsupported semantics therefore fail with a
// useful diagnostic instead of reaching a decode loop.
void validate_moe_backend_capabilities(const CompiledModelProgram& program,
                                       std::string_view backend,
                                       MoeBackendCapabilities capabilities);

CompiledModelProgram build_model_program(const ResolvedModel& model);

} // namespace celeg
