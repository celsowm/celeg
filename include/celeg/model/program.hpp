#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace celeg {

enum class CompiledMixer : unsigned char {
    Attention, ShortConvolution, GatedDeltaNet, Mamba2, MlpOnly
};
enum class CompiledFeedForward : unsigned char { Dense, MixtureOfExperts, None };

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

struct CompiledDenseFeedForwardProgram {
    int intermediate_size = 0;
    ActivationKind activation = ActivationKind::SwiGLU;

    void validate() const;
};

using CompiledFeedForwardVariant = std::variant<
    std::monostate,
    CompiledDenseFeedForwardProgram,
    MoeLayerProgram>;

// The variant is the only stored post-mixer feed-forward identity. The legacy
// enum remains only as a derived compatibility tag, never as independent state.
class CompiledFeedForwardProgram : public CompiledFeedForwardVariant {
public:
    using CompiledFeedForwardVariant::operator=;

    CompiledFeedForwardProgram();
    CompiledFeedForwardProgram(const CompiledFeedForwardProgram&) = default;
    CompiledFeedForwardProgram(CompiledFeedForwardProgram&&) = default;
    CompiledFeedForwardProgram& operator=(const CompiledFeedForwardProgram&) = default;
    CompiledFeedForwardProgram& operator=(CompiledFeedForwardProgram&&) = default;
    CompiledFeedForwardProgram& operator=(CompiledFeedForward kind);

    CompiledFeedForwardVariant& storage() { return *this; }
    const CompiledFeedForwardVariant& storage() const { return *this; }
    bool enabled() const { return !std::holds_alternative<std::monostate>(storage()); }
    void validate() const;
};

CompiledFeedForward compiled_feed_forward_kind(const CompiledFeedForwardProgram& program);
bool operator==(const CompiledFeedForwardProgram& program, CompiledFeedForward kind);
bool operator==(CompiledFeedForward kind, const CompiledFeedForwardProgram& program);
bool operator!=(const CompiledFeedForwardProgram& program, CompiledFeedForward kind);
bool operator!=(CompiledFeedForward kind, const CompiledFeedForwardProgram& program);

class CompiledFeedForwardEnabledView {
public:
    CompiledFeedForwardEnabledView() = default;
    void bind(CompiledFeedForwardProgram& program) { program_ = &program; }
    operator bool() const { return program_ && program_->enabled(); }
    CompiledFeedForwardEnabledView& operator=(bool enabled) {
        if (!program_) throw std::logic_error("compiled feed-forward enabled view is not bound");
        if (!enabled) {
            program_->storage() = std::monostate{};
        } else if (!program_->enabled()) {
            program_->storage() = CompiledDenseFeedForwardProgram{};
        }
        return *this;
    }

private:
    CompiledFeedForwardProgram* program_ = nullptr;
};

class CompiledFeedForwardIntermediateView {
public:
    using Fallback = int (*)(const void*);

    CompiledFeedForwardIntermediateView() = default;
    void bind(CompiledFeedForwardProgram& program,
              const void* fallback_context = nullptr,
              Fallback fallback = nullptr) {
        program_ = &program;
        fallback_context_ = fallback_context;
        fallback_ = fallback;
    }
    int value() const {
        if (!program_) return 0;
        if (const auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(&program_->storage())) {
            return dense->intermediate_size;
        }
        if (const auto* moe = std::get_if<MoeLayerProgram>(&program_->storage())) {
            return moe->routed.mlp.intermediate_size;
        }
        return fallback_ ? fallback_(fallback_context_) : 0;
    }
    operator int() const { return value(); }
    CompiledFeedForwardIntermediateView& operator=(int intermediate_size) {
        if (!program_) throw std::logic_error("compiled feed-forward width view is not bound");
        if (auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(&program_->storage())) {
            dense->intermediate_size = intermediate_size;
        } else if (auto* moe = std::get_if<MoeLayerProgram>(&program_->storage())) {
            moe->routed.mlp.intermediate_size = intermediate_size;
        } else {
            program_->storage() = CompiledDenseFeedForwardProgram{
                intermediate_size, ActivationKind::SwiGLU};
        }
        return *this;
    }

private:
    CompiledFeedForwardProgram* program_ = nullptr;
    const void* fallback_context_ = nullptr;
    Fallback fallback_ = nullptr;
};

class CompiledFeedForwardActivationView {
public:
    using Fallback = ActivationKind (*)(const void*);

    CompiledFeedForwardActivationView() = default;
    void bind(CompiledFeedForwardProgram& program,
              const void* fallback_context = nullptr,
              Fallback fallback = nullptr) {
        program_ = &program;
        fallback_context_ = fallback_context;
        fallback_ = fallback;
    }
    ActivationKind value() const {
        if (program_) {
            if (const auto* dense =
                    std::get_if<CompiledDenseFeedForwardProgram>(&program_->storage())) {
                return dense->activation;
            }
        }
        return fallback_ ? fallback_(fallback_context_) : ActivationKind::SwiGLU;
    }
    operator ActivationKind() const { return value(); }
    CompiledFeedForwardActivationView& operator=(ActivationKind activation) {
        if (!program_) throw std::logic_error("compiled feed-forward activation view is not bound");
        if (auto* dense = std::get_if<CompiledDenseFeedForwardProgram>(&program_->storage())) {
            dense->activation = activation;
        } else if (std::holds_alternative<std::monostate>(program_->storage())) {
            program_->storage() = CompiledDenseFeedForwardProgram{0, activation};
        } else {
            throw std::logic_error("MoE activation is part of the MoE semantic program");
        }
        return *this;
    }

private:
    CompiledFeedForwardProgram* program_ = nullptr;
    const void* fallback_context_ = nullptr;
    Fallback fallback_ = nullptr;
};

class CompiledMoeView {
public:
    CompiledMoeView() = default;
    void bind(CompiledFeedForwardProgram& program) { program_ = &program; }
    MoeLayerProgram* get() {
        return program_ ? std::get_if<MoeLayerProgram>(&program_->storage()) : nullptr;
    }
    const MoeLayerProgram* get() const {
        return program_ ? std::get_if<MoeLayerProgram>(&program_->storage()) : nullptr;
    }
    MoeLayerProgram* operator()() { return get(); }
    const MoeLayerProgram* operator()() const { return get(); }
    bool has_value() const { return get() != nullptr; }
    explicit operator bool() const { return has_value(); }
    MoeLayerProgram& value() {
        MoeLayerProgram* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    const MoeLayerProgram& value() const {
        const MoeLayerProgram* result = get();
        if (!result) throw std::bad_optional_access();
        return *result;
    }
    MoeLayerProgram* operator->() { return &value(); }
    const MoeLayerProgram* operator->() const { return &value(); }
    MoeLayerProgram& operator*() { return value(); }
    const MoeLayerProgram& operator*() const { return value(); }
    CompiledMoeView& operator=(const MoeLayerProgram& value) {
        if (!program_) throw std::logic_error("compiled MoE view is not bound");
        program_->storage() = value;
        return *this;
    }
    CompiledMoeView& operator=(MoeLayerProgram&& value) {
        if (!program_) throw std::logic_error("compiled MoE view is not bound");
        program_->storage() = std::move(value);
        return *this;
    }
    void reset() {
        if (!program_) throw std::logic_error("compiled MoE view is not bound");
        program_->storage() = std::monostate{};
    }

private:
    CompiledFeedForwardProgram* program_ = nullptr;
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

using CompiledMixerVariant = std::variant<
    CompiledAttentionProgram,
    ShortConvolutionSpec,
    GatedDeltaNetSpec,
    Mamba2Spec,
    MlpBlockSpec>;

// The variant is the only stored mixer identity. Assignment from the legacy
// CompiledMixer enum is intentionally factory-like: it selects a default
// alternative but does not store a second tag.
class CompiledMixerProgram : public CompiledMixerVariant {
public:
    using CompiledMixerVariant::CompiledMixerVariant;
    using CompiledMixerVariant::operator=;

    CompiledMixerProgram() = default;
    CompiledMixerProgram(const CompiledMixerProgram&) = default;
    CompiledMixerProgram(CompiledMixerProgram&&) = default;
    CompiledMixerProgram& operator=(const CompiledMixerProgram&) = default;
    CompiledMixerProgram& operator=(CompiledMixerProgram&&) = default;
    CompiledMixerProgram& operator=(CompiledMixer kind);

    CompiledMixerVariant& storage() { return *this; }
    const CompiledMixerVariant& storage() const { return *this; }
};

CompiledMixer compiled_mixer_kind(const CompiledMixerProgram& mixer);
bool operator==(const CompiledMixerProgram& mixer, CompiledMixer kind);
bool operator==(CompiledMixer kind, const CompiledMixerProgram& mixer);
bool operator!=(const CompiledMixerProgram& mixer, CompiledMixer kind);
bool operator!=(CompiledMixer kind, const CompiledMixerProgram& mixer);

// Optional-like views preserve the compact consumer API without duplicating
// semantic state. They only point into CompiledLayerProgram::mixer.
template <typename T>
class CompiledMixerView {
public:
    CompiledMixerView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    T* get() {
        return mixer_ ? std::get_if<T>(&mixer_->storage()) : nullptr;
    }
    const T* get() const {
        return mixer_ ? std::get_if<T>(&mixer_->storage()) : nullptr;
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

    CompiledMixerView& operator=(const T& value) {
        if (!mixer_) throw std::logic_error("compiled mixer view is not bound");
        mixer_->storage() = value;
        return *this;
    }
    CompiledMixerView& operator=(T&& value) {
        if (!mixer_) throw std::logic_error("compiled mixer view is not bound");
        mixer_->storage() = std::move(value);
        return *this;
    }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

class CompiledAttentionView {
public:
    CompiledAttentionView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    AttentionSpec* get() {
        auto* result = mixer_
            ? std::get_if<CompiledAttentionProgram>(&mixer_->storage()) : nullptr;
        return result ? &result->semantics : nullptr;
    }
    const AttentionSpec* get() const {
        const auto* result = mixer_
            ? std::get_if<CompiledAttentionProgram>(&mixer_->storage()) : nullptr;
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

    CompiledAttentionView& operator=(const AttentionSpec& semantics) {
        if (!mixer_) throw std::logic_error("compiled attention view is not bound");
        if (auto* current = get()) {
            *current = semantics;
        } else {
            mixer_->storage() = CompiledAttentionProgram{semantics, {}};
        }
        return *this;
    }
    CompiledAttentionView& operator=(AttentionSpec&& semantics) {
        if (!mixer_) throw std::logic_error("compiled attention view is not bound");
        if (auto* current = get()) {
            *current = std::move(semantics);
        } else {
            mixer_->storage() = CompiledAttentionProgram{std::move(semantics), {}};
        }
        return *this;
    }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

class CompiledAttentionStateLayoutView {
public:
    CompiledAttentionStateLayoutView() = default;

    void bind(CompiledMixerProgram& mixer) { mixer_ = &mixer; }

    CompiledAttentionStateLayout* get() {
        auto* result = mixer_
            ? std::get_if<CompiledAttentionProgram>(&mixer_->storage()) : nullptr;
        return result ? &result->state_layout : nullptr;
    }
    const CompiledAttentionStateLayout* get() const {
        const auto* result = mixer_
            ? std::get_if<CompiledAttentionProgram>(&mixer_->storage()) : nullptr;
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

    CompiledAttentionStateLayoutView& operator=(
        const CompiledAttentionStateLayout& layout) {
        if (!mixer_) throw std::logic_error("compiled state-layout view is not bound");
        if (auto* current = get()) {
            *current = layout;
        } else {
            mixer_->storage() = CompiledAttentionProgram{{}, layout};
        }
        return *this;
    }
    CompiledAttentionStateLayoutView& operator=(
        CompiledAttentionStateLayout&& layout) {
        if (!mixer_) throw std::logic_error("compiled state-layout view is not bound");
        if (auto* current = get()) {
            *current = std::move(layout);
        } else {
            mixer_->storage() = CompiledAttentionProgram{{}, std::move(layout)};
        }
        return *this;
    }

private:
    CompiledMixerProgram* mixer_ = nullptr;
};

enum class CompiledChunkCapability : uint8_t {
    Native,
    SequentialAdapter,
    Unsupported,
};

// Immutable execution description produced before a backend starts serving.
// Mixer and post-mixer feed-forward variants are the semantic sources of truth;
// compatibility members below are non-owning views into those variants.
struct CompiledLayerProgram {
    CompiledMixerProgram mixer;
    CompiledAttentionView attention;
    CompiledAttentionStateLayoutView state_layout;
    CompiledMixerView<ShortConvolutionSpec> short_convolution;
    CompiledMixerView<GatedDeltaNetSpec> gated_delta_net;
    CompiledMixerView<Mamba2Spec> mamba2;
    CompiledMixerView<MlpBlockSpec> mlp_only;

    CompiledFeedForwardProgram feed_forward;
    CompiledFeedForwardEnabledView execute_feed_forward;
    CompiledFeedForwardIntermediateView feed_forward_intermediate;
    CompiledFeedForwardActivationView feed_forward_activation;
    CompiledMoeView moe;

    CompiledChunkCapability chunk_capability = CompiledChunkCapability::Native;
    std::vector<std::size_t> weight_request_indices;
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
    CompiledFeedForward feed_forward_kind() const {
        return compiled_feed_forward_kind(feed_forward);
    }

private:
    void bind_mixer_views();
    void bind_feed_forward_views();
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