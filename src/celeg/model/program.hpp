#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace celeg {

enum class MoeRouterScoreKind : unsigned char {
    SigmoidProbabilities,
    SoftmaxLogits,
};
enum class MoeNormalizationKind : unsigned char { None, SumSelected };
enum class MoePayloadLayout : unsigned char { Individual, Stacked, Fused };
enum class MoeActivation : unsigned char { SwiGLU, GeluTanh };

struct ExpertPayloadRegion {
    TensorRole role = TensorRole::MoeExpertGate;
    std::size_t elements = 0;
};

struct ExpertPayloadSchema {
    MoePayloadLayout layout = MoePayloadLayout::Individual;
    std::vector<ExpertPayloadRegion> regions;

    void validate() const;
    std::size_t elements() const;
    std::string fingerprint() const;
};

struct RouterProgram {
    MoeRouterScoreKind score = MoeRouterScoreKind::SigmoidProbabilities;
    MoeSelectionSpec selection = MoeTopKSelectionSpec{};
    MoeNormalizationKind normalization = MoeNormalizationKind::None;
    int expert_count = 0;
    int experts_per_token = 0;
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
    MoeCombineOrder combine_order = MoeCombineOrder::RoutedThenShared;
};

struct MoeLayerProgram {
    RouterProgram router;
    RoutedExpertProgram routed;
    std::optional<SharedExpertProgram> shared;

    void validate() const;
    std::string fingerprint() const;
};

struct CompiledDenseFeedForwardProgram {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::SwiGLU;

    void validate() const;
};

using CompiledFeedForwardProgram = std::variant<
    std::monostate,
    CompiledDenseFeedForwardProgram,
    MoeLayerProgram>;

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

inline std::size_t compiled_state_scalar_bytes(StateScalarType scalar) {
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

struct CompiledOrdinaryKvStateLayout {
    int key_width = 0;
    int value_width = 0;
    OrdinaryKvStorageSpec storage;

    std::size_t persistent_elements() const {
        return static_cast<std::size_t>(key_width) +
               static_cast<std::size_t>(value_width);
    }
    std::size_t persistent_bytes() const {
        return static_cast<std::size_t>(key_width) *
                   compiled_state_scalar_bytes(storage.key) +
               static_cast<std::size_t>(value_width) *
                   compiled_state_scalar_bytes(storage.value);
    }
    void validate() const;
};

struct CompiledLatentStateLayout {
    int latent_width = 0;
    int rotary_width = 0;
    LatentStorageSpec storage;

    std::size_t persistent_elements() const {
        return static_cast<std::size_t>(latent_width) +
               static_cast<std::size_t>(rotary_width);
    }
    std::size_t persistent_bytes() const {
        return static_cast<std::size_t>(latent_width) *
                   compiled_state_scalar_bytes(storage.latent) +
               static_cast<std::size_t>(rotary_width) *
                   compiled_state_scalar_bytes(storage.rotary);
    }
    void validate() const;
};

using CompiledAttentionStateLayout = std::variant<
    CompiledOrdinaryKvStateLayout,
    CompiledLatentStateLayout>;

enum class AttentionExecutionKind : std::uint8_t {
    Standard,
    Latent,
    FactorizedLatent,
};

struct CompiledAttentionExecution {
    AttentionExecutionKind kind = AttentionExecutionKind::Standard;
    bool has_key_value = true;
    bool has_query_key_norm = false;
    bool has_rope = false;
    bool has_decoupled_rope = false;
    RopePairingKind rope_pairing = RopePairingKind::SplitHalf;
    AttentionGateGranularity gate_granularity = AttentionGateGranularity::OutputWise;
    int rotary_width = 0;

    void validate() const;
};

struct CompiledAttentionProgram {
    AttentionSpec semantics;
    CompiledAttentionStateLayout state_layout;
    CompiledAttentionExecution execution;
};

using CompiledMixerProgram = std::variant<
    CompiledAttentionProgram,
    ShortConvolutionSpec,
    GatedDeltaNetSpec,
    Mamba2Spec,
    MlpBlockSpec>;

struct CompiledLayerProgram {
    CompiledMixerProgram mixer;
    CompiledFeedForwardProgram feed_forward;
    std::vector<std::size_t> weight_request_indices;
    SublayerNormSpec mixer_norm;
    SublayerNormSpec feed_forward_norm;
    ResidualSpec residual;
};

struct CompiledModelProgram {
    int hidden = 0;
    std::string identity;
    std::size_t weight_request_count = 0;
    std::vector<CompiledLayerProgram> layers;
    std::vector<int> norm_after_layers;
    std::vector<std::size_t> unlayered_weight_request_indices;
    PerLayerInputPlan per_layer_input;
    NormSpec final_norm;
    ModelGraph::EmbeddingTransformSpec embedding_transform;
    float logits_multiplier = 1.0f;
    float logits_divisor = 1.0f;
    float final_logit_softcap = 0.0f;
    std::string semantic_fingerprint;

    bool has_moe() const;

    const CompiledAttentionProgram* last_attention() const noexcept {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            if (const auto* attention = std::get_if<CompiledAttentionProgram>(&it->mixer)) {
                return attention;
            }
        }
        return nullptr;
    }

    const MoeLayerProgram* first_moe() const noexcept {
        for (const CompiledLayerProgram& layer : layers) {
            if (const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward)) {
                return moe;
            }
        }
        return nullptr;
    }

    const CompiledDenseFeedForwardProgram* first_dense_feed_forward() const noexcept {
        for (const CompiledLayerProgram& layer : layers) {
            if (const auto* dense =
                    std::get_if<CompiledDenseFeedForwardProgram>(&layer.feed_forward)) {
                return dense;
            }
        }
        return nullptr;
    }

    void validate() const;
};

CompiledModelProgram build_model_program(const ResolvedModel& model);

}
