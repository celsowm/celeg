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

enum class CompiledMixer : unsigned char {
    Attention, ShortConvolution, GatedDeltaNet, Mamba2, MlpOnly
};
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts };

// Neutral, immutable MoE semantics. These values are intentionally free of
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
    int groups_per_token = 0;
    int group_score_top_k = 0;
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

struct CompiledAttentionProgram {
    AttentionSpec semantics;
    CompiledAttentionStateLayout state_layout;
};

using CompiledMixerProgram = std::variant<
    CompiledAttentionProgram,
    ShortConvolutionSpec,
    GatedDeltaNetSpec,
    Mamba2Spec,
    MlpBlockSpec>;

CompiledMixer compiled_mixer_kind(const CompiledMixerProgram& mixer);
bool operator==(const CompiledMixerProgram& mixer, CompiledMixer kind);
bool operator==(CompiledMixer kind, const CompiledMixerProgram& mixer);
bool operator!=(const CompiledMixerProgram& mixer, CompiledMixer kind);
bool operator!=(CompiledMixer kind, const CompiledMixerProgram& mixer);

// Optional-like views preserve the compact consumer API without duplicating
// semantic state. They only point into CompiledLayerProgram::mixer, so changing
// the active alternative is the only way to change mixer identity.
template <typename T>
class CompiledMixerView {
public:
    CompiledMixerView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    T* get() {
        return mixer_ ? std::get_if<T>(mixer_) : nullptr;
    }
    const T* get() const {
        return mixer_ ? std::get_if<T>(mixer_) : nullptr;
    }
    T* operator()() { return get(); }
    const T* operator()() const { return get(); }
    bool has_value() const { return get() != nullptr; }
    explicit operator bool() const { return has_value(); }
    T& value() {
        T* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    const T& value() const {
        const T* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    T* operator->() { return &value(); }
    const T* operator->() const { return &value(); }
    T& operator*() { return value(); }
    const T& operator*() const { return value(); }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

class CompiledAttentionView {
public:
    CompiledAttentionView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    AttentionSpec* get() {
        auto* result = mixer_ ? std::get_if<CompiledAttentionProgram>(mixer_) : nullptr;
        return result ? &result->semantics : nullptr;
    }
    const AttentionSpec* get() const {
        const auto* result = mixer_ ? std::get_if<CompiledAttentionProgram>(mixer_) : nullptr;
        return result ? &result->semantics : nullptr;
    }
    AttentionSpec* operator()() { return get(); }
    const AttentionSpec* operator()() const { return get(); }
    bool has_value() const { return get() != nullptr; }
    explicit operator bool() const { return has_value(); }
    AttentionSpec& value() {
        AttentionSpec* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    const AttentionSpec& value() const {
        const AttentionSpec* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    AttentionSpec* operator->() { return &value(); }
    const AttentionSpec* operator->() const { return &value(); }
    AttentionSpec& operator*() { return value(); }
    const AttentionSpec& operator*() const { return value(); }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

class CompiledAttentionStateLayoutView {
public:
    CompiledAttentionStateLayoutView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    CompiledAttentionStateLayout* get() {
        auto* result = mixer_ ? std::get_if<CompiledAttentionProgram>(mixer_) : nullptr;
        return result ? &result->state_layout : nullptr;
    }
    const CompiledAttentionStateLayout* get() const {
        const auto* result = mixer_ ? std::get_if<CompiledAttentionProgram>(mixer_) : nullptr;
        return result ? &result->state_layout : nullptr;
    }
    CompiledAttentionStateLayout* operator()() { return get(); }
    const CompiledAttentionStateLayout* operator()() const { return get(); }
    bool has_value() const { return get() != nullptr; }
    explicit operator bool() const { return has_value(); }
    CompiledAttentionStateLayout& value() {
        CompiledAttentionStateLayout* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    const CompiledAttentionStateLayout& value() const {
        const CompiledAttentionStateLayout* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    CompiledAttentionStateLayout* operator->() { return &value(); }
    const CompiledAttentionStateLayout* operator->() const { return &value(); }
    CompiledAttentionStateLayout& operator*() { return value(); }
    const CompiledAttentionStateLayout& operator*() const { return value(); }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

enum class CompiledChunkCapability : uint8_t {
    Native,
    SequentialAdapter,
    Unsupported,
};

// Immutable execution description produced before a backend starts serving.
// `mixer` is the single semantic source of truth. The optional-like members
// below are non-owning views into that variant and therefore cannot create
// contradictory mixer states.
struct CompiledLayerProgram {
    CompiledMixerProgram mixer;
    CompiledAttentionView attention;
    CompiledAttentionStateLayoutView state_layout;
    CompiledMixerView<ShortConvolutionSpec> short_convolution;
    CompiledMixerView<GatedDeltaNetSpec> gated_delta_net;
    CompiledMixerView<Mamba2Spec> mamba2;
    CompiledMixerView<MlpBlockSpec> mlp_only;

    CompiledFeedForward feed_forward = CompiledFeedForward::Dense;
    bool execute_feed_forward = true;
    CompiledChunkCapability chunk_capability = CompiledChunkCapability::Native;
    int feed_forward_intermediate = 0;
    ActivationKind feed_forward_activation = ActivationKind::SwiGLU;
    std::vector<std::size_t> weight_request_indices;
    std::optional<MoeLayerProgram> moe;
    NormSpec operator_norm;
    NormSpec post_attention_norm;
    NormSpec feed_forward_norm;
    NormSpec post_feed_forward_norm;
    ResidualSpec residual;

    CompiledLayerProgram();
    CompiledLayerProgram(const CompiledLayerProgram& other);
    CompiledLayerProgram(CompiledLayerProgram&& other);
    CompiledLayerProgram& operator=(const CompiledLayerProgram& other);
    CompiledLayerProgram& operator=(CompiledLayerProgram&& other);

    CompiledMixer mixer_kind() const { return compiled_mixer_kind(mixer); }

private:
    void bind_mixer_views();
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
    void validate() const;
};

CompiledModelProgram build_model_program(const ResolvedModel& model);

} // namespace celeg
