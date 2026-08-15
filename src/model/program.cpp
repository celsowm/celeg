#include "celeg/model/program.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

namespace {

constexpr float kPerLayerResidualScale = 0.7071067811865475f;

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(message);
    }
    return left * right;
}

template <typename T>
void append_field(std::ostringstream& out, T value) {
    out << static_cast<int>(value) << ';';
}

void append_field(std::ostringstream& out, int value) { out << value << ';'; }
void append_field(std::ostringstream& out, std::size_t value) { out << value << ';'; }
void append_field(std::ostringstream& out, float value) { out << value << ';'; }
void append_field(std::ostringstream& out, bool value) { out << (value ? 1 : 0) << ';'; }

std::string fingerprint_text(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

} // namespace

void ExpertPayloadSchema::validate() const {
    for (const ExpertPayloadRegion& region : regions) {
        if (region.elements == 0) {
            throw std::invalid_argument("MoE payload region has no elements");
        }
    }
}

std::string ExpertPayloadSchema::fingerprint() const {
    std::ostringstream out;
    append_field(out, layout);
    for (const auto& region : regions) {
        append_field(out, region.role);
        append_field(out, region.elements);
    }
    return fingerprint_text(out.str());
}

void RouterProgram::validate() const {
    if (expert_count <= 0 || experts_per_token <= 0 ||
        experts_per_token > expert_count || !std::isfinite(routed_scaling) ||
        routed_scaling == 0.0f) {
        throw std::invalid_argument("invalid compiled MoE router program");
    }
    if (selection == MoeSelectionKind::GroupedTopK &&
        (group_count <= 0 || experts_per_group <= 0 ||
         group_count * experts_per_group != expert_count ||
         groups_per_token <= 0 || groups_per_token > group_count ||
         group_score_top_k <= 0 || group_score_top_k > experts_per_group)) {
        throw std::invalid_argument("invalid grouped MoE selection program");
    }
    if (selection == MoeSelectionKind::TopK &&
        (group_count != 0 || experts_per_group != 0)) {
        throw std::invalid_argument("ordinary MoE selection cannot have groups");
    }
}

std::string RouterProgram::fingerprint() const {
    std::ostringstream out;
    append_field(out, score);
    append_field(out, selection);
    append_field(out, normalization);
    append_field(out, expert_count);
    append_field(out, experts_per_token);
    append_field(out, group_count);
    append_field(out, experts_per_group);
    append_field(out, groups_per_token);
    append_field(out, group_score_top_k);
    append_field(out, has_expert_bias);
    append_field(out, routed_scaling);
    return fingerprint_text(out.str());
}

void ExpertMlpProgram::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("invalid compiled MoE expert MLP program");
    }
}

void CompiledAttentionStateLayout::validate() const {
    if (key_width < 0 || value_width < 0 || latent_width < 0 || rotary_width < 0 ||
        persistent_elements == 0 ||
        persistent_elements != key_elements + value_elements +
            latent_elements + rotary_elements) {
        throw std::invalid_argument("invalid compiled attention state layout");
    }
    if (kind == CompiledStateLayoutKind::OrdinaryKv &&
        (key_width <= 0 || value_width <= 0 || latent_width != 0 || rotary_width != 0 ||
         key_elements == 0 || value_elements == 0 || latent_elements != 0 ||
         rotary_elements != 0)) {
        throw std::invalid_argument("ordinary KV state layout has invalid widths");
    }
    if (kind == CompiledStateLayoutKind::Latent &&
        (latent_width <= 0 || latent_elements == 0)) {
        throw std::invalid_argument("latent state layout has no persistent latent width");
    }
    (void)scalar_bytes(storage.key);
    (void)scalar_bytes(storage.value);
    (void)scalar_bytes(storage.latent);
    (void)scalar_bytes(storage.rotary);
    (void)scalar_bytes(storage.recurrent);
    if (persistent_bytes() == 0) {
        throw std::invalid_argument("compiled attention state has zero byte size");
    }
}

std::string ExpertMlpProgram::fingerprint() const {
    std::ostringstream out;
    append_field(out, activation);
    append_field(out, hidden_size);
    append_field(out, intermediate_size);
    return fingerprint_text(out.str());
}

void MoeLayerProgram::validate() const {
    router.validate();
    routed.mlp.validate();
    routed.payload.validate();
    if (routed.payload.regions.empty()) {
        throw std::invalid_argument("MoE program has no expert payload regions");
    }
    if (shared) shared->mlp.validate();
    if (output.has_shared_expert != shared.has_value()) {
        throw std::invalid_argument("MoE shared-output flag does not match program");
    }
    if (residency.expert_count != router.expert_count) {
        throw std::invalid_argument("MoE residency expert count mismatch");
    }
}

std::string MoeLayerProgram::fingerprint() const {
    std::ostringstream out;
    out << router.fingerprint() << ';' << routed.mlp.fingerprint() << ';'
        << routed.payload.fingerprint() << ';';
    if (shared) out << shared->mlp.fingerprint();
    out << ';';
    append_field(out, output.has_shared_expert);
    append_field(out, output.combine_order);
    append_field(out, residency.expert_count);
    append_field(out, residency.payload_elements);
    return fingerprint_text(out.str());
}

void CompiledDenseFeedForwardProgram::validate() const {
    if (intermediate_size <= 0) {
        throw std::invalid_argument("compiled dense feed-forward has no width");
    }
}

CompiledFeedForwardProgram::CompiledFeedForwardProgram()
    : CompiledFeedForwardVariant(CompiledDenseFeedForwardProgram{}) {}

CompiledFeedForwardProgram& CompiledFeedForwardProgram::operator=(
    CompiledFeedForward kind) {
    switch (kind) {
    case CompiledFeedForward::Dense:
        if (!std::holds_alternative<CompiledDenseFeedForwardProgram>(storage())) {
            storage() = CompiledDenseFeedForwardProgram{};
        }
        break;
    case CompiledFeedForward::MixtureOfExperts:
        if (!std::holds_alternative<MoeLayerProgram>(storage())) {
            storage() = MoeLayerProgram{};
        }
        break;
    case CompiledFeedForward::None:
        storage() = std::monostate{};
        break;
    }
    return *this;
}

CompiledFeedForward compiled_feed_forward_kind(
    const CompiledFeedForwardProgram& program) {
    return std::visit([](const auto& value) {
        using FeedForward = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<FeedForward, std::monostate>) {
            return CompiledFeedForward::None;
        } else if constexpr (
            std::is_same_v<FeedForward, CompiledDenseFeedForwardProgram>) {
            return CompiledFeedForward::Dense;
        } else if constexpr (std::is_same_v<FeedForward, MoeLayerProgram>) {
            return CompiledFeedForward::MixtureOfExperts;
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled compiled feed-forward variant");
        }
    }, program.storage());
}

void CompiledFeedForwardProgram::validate() const {
    std::visit([](const auto& value) {
        using FeedForward = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<FeedForward, std::monostate>) {
            return;
        } else if constexpr (
            std::is_same_v<FeedForward, CompiledDenseFeedForwardProgram> ||
            std::is_same_v<FeedForward, MoeLayerProgram>) {
            value.validate();
        } else {
            static_assert(always_false_v<FeedForward>,
                          "unhandled compiled feed-forward validation variant");
        }
    }, storage());
}

bool operator==(const CompiledFeedForwardProgram& program,
                CompiledFeedForward kind) {
    return compiled_feed_forward_kind(program) == kind;
}

bool operator==(CompiledFeedForward kind,
                const CompiledFeedForwardProgram& program) {
    return program == kind;
}

bool operator!=(const CompiledFeedForwardProgram& program,
                CompiledFeedForward kind) {
    return !(program == kind);
}

bool operator!=(CompiledFeedForward kind,
                const CompiledFeedForwardProgram& program) {
    return !(program == kind);
}

PerLayerInputPlan PerLayerInputPlan::derive(const ResolvedModel& model) {
    PerLayerInputPlan result;
    const ModelGraph& graph = model.graph;
    const auto enabled_layer = std::find_if(
        graph.layers.begin(), graph.layers.end(),
        [](const LayerSpec& layer) { return layer.per_layer_input.enabled; });
    if (enabled_layer == graph.layers.end()) {
        return result;
    }
    if (graph.layers.empty() || graph.hidden <= 0 ||
        enabled_layer->per_layer_input.input_size <= 0) {
        throw std::invalid_argument("invalid per-layer input dimensions");
    }
    result.enabled = true;
    result.layer_count = static_cast<int>(graph.layers.size());
    result.input_size = enabled_layer->per_layer_input.input_size;
    result.packed_width = checked_product(
        static_cast<std::size_t>(result.layer_count),
        static_cast<std::size_t>(result.input_size),
        "per-layer input width overflows size_t");
    if (result.packed_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "per-layer input width exceeds the compiled execution index range");
    }
    result.token_scale = std::sqrt(static_cast<float>(result.input_size));
    result.context_scale = 1.0f / std::sqrt(static_cast<float>(graph.hidden));
    result.residual_scale = kPerLayerResidualScale;
    if (model.graph.layers.empty()) {
        throw std::invalid_argument("per-layer input requires resolved layer specifications");
    }
    result.activation = model.graph.layers.front().per_layer_input.activation;
    result.norm_epsilon = model.graph.layers.front().per_layer_input_norm.epsilon;
    for (const LayerSpec& layer : model.graph.layers) {
        if (!layer.per_layer_input.enabled ||
            layer.per_layer_input.input_size != result.input_size ||
            layer.per_layer_input.activation != result.activation ||
            layer.per_layer_input_norm.epsilon != result.norm_epsilon) {
            throw std::invalid_argument("inconsistent per-layer input specification");
        }
    }
    result.validate();
    return result;
}

std::size_t PerLayerInputPlan::checked_elements(std::size_t rows) const {
    if (!enabled) return 0;
    return checked_product(rows, packed_width,
                           "per-layer input row product overflows size_t");
}

void PerLayerInputPlan::validate() const {
    if (!enabled) return;
    if (layer_count <= 0 || input_size <= 0 || packed_width == 0 ||
        packed_width != checked_product(static_cast<std::size_t>(layer_count),
                                        static_cast<std::size_t>(input_size),
                                        "per-layer input width overflows size_t") ||
        !std::isfinite(token_scale) || !std::isfinite(context_scale) ||
        !std::isfinite(residual_scale) || !(norm_epsilon > 0.0f) ||
        !std::isfinite(norm_epsilon)) {
        throw std::invalid_argument("invalid compiled per-layer input plan");
    }
}

CompiledMixerProgram& CompiledMixerProgram::operator=(CompiledMixer kind) {
    switch (kind) {
    case CompiledMixer::Attention:
        storage() = CompiledAttentionProgram{};
        break;
    case CompiledMixer::ShortConvolution:
        storage() = ShortConvolutionSpec{};
        break;
    case CompiledMixer::GatedDeltaNet:
        storage() = GatedDeltaNetSpec{};
        break;
    case CompiledMixer::Mamba2:
        storage() = Mamba2Spec{};
        break;
    case CompiledMixer::MlpOnly:
        storage() = MlpBlockSpec{};
        break;
    }
    return *this;
}

CompiledMixer compiled_mixer_kind(const CompiledMixerProgram& mixer) {
    return std::visit([](const auto& value) {
        using Mixer = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Mixer, CompiledAttentionProgram>) {
            return CompiledMixer::Attention;
        } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
            return CompiledMixer::ShortConvolution;
        } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
            return CompiledMixer::GatedDeltaNet;
        } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
            return CompiledMixer::Mamba2;
        } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
            return CompiledMixer::MlpOnly;
        } else {
            static_assert(always_false_v<Mixer>, "unhandled compiled mixer variant");
        }
    }, mixer.storage());
}

bool operator==(const CompiledMixerProgram& mixer, CompiledMixer kind) {
    return compiled_mixer_kind(mixer) == kind;
}

bool operator==(CompiledMixer kind, const CompiledMixerProgram& mixer) {
    return mixer == kind;
}

bool operator!=(const CompiledMixerProgram& mixer, CompiledMixer kind) {
    return !(mixer == kind);
}

bool operator!=(CompiledMixer kind, const CompiledMixerProgram& mixer) {
    return !(mixer == kind);
}

void CompiledLayerProgram::bind_mixer_views() {
    attention.bind(mixer);
    state_layout.bind(mixer);
    short_convolution.bind(mixer);
    gated_delta_net.bind(mixer);
    mamba2.bind(mixer);
    mlp_only.bind(mixer);
}

void CompiledLayerProgram::bind_feed_forward_views() {
    execute_feed_forward.bind(feed_forward);
    feed_forward_intermediate.bind(
        feed_forward, this, [](const void* context) {
            const auto& layer =
                *static_cast<const CompiledLayerProgram*>(context);
            const MlpBlockSpec* mlp = layer.mlp_only();
            return mlp ? mlp->intermediate_size : 0;
        });
    feed_forward_activation.bind(
        feed_forward, this, [](const void* context) {
            const auto& layer =
                *static_cast<const CompiledLayerProgram*>(context);
            const MlpBlockSpec* mlp = layer.mlp_only();
            return mlp ? mlp->activation : ActivationKind::SwiGLU;
        });
    moe.bind(feed_forward);
}

CompiledLayerProgram::CompiledLayerProgram() {
    bind_mixer_views();
    bind_feed_forward_views();
}

CompiledLayerProgram::CompiledLayerProgram(const CompiledLayerProgram& other)
    : CompiledLayerProgram() {
    *this = other;
}

CompiledLayerProgram::CompiledLayerProgram(CompiledLayerProgram&& other)
    : CompiledLayerProgram() {
    *this = std::move(other);
}

CompiledLayerProgram& CompiledLayerProgram::operator=(
    const CompiledLayerProgram& other) {
    if (this == &other) return *this;
    mixer = other.mixer;
    feed_forward = other.feed_forward;
    chunk_capability = other.chunk_capability;
    weight_request_indices = other.weight_request_indices;
    operator_norm = other.operator_norm;
    post_attention_norm = other.post_attention_norm;
    feed_forward_norm = other.feed_forward_norm;
    post_feed_forward_norm = other.post_feed_forward_norm;
    residual = other.residual;
    return *this;
}

CompiledLayerProgram& CompiledLayerProgram::operator=(
    CompiledLayerProgram&& other) {
    if (this == &other) return *this;
    mixer = std::move(other.mixer);
    feed_forward = std::move(other.feed_forward);
    chunk_capability = other.chunk_capability;
    weight_request_indices = std::move(other.weight_request_indices);
    operator_norm = std::move(other.operator_norm);
    post_attention_norm = std::move(other.post_attention_norm);
    feed_forward_norm = std::move(other.feed_forward_norm);
    post_feed_forward_norm = std::move(other.post_feed_forward_norm);
    residual = std::move(other.residual);
    return *this;
}

bool CompiledModelProgram::has_moe() const {
    for (const auto& layer : layers) {
        if (std::holds_alternative<MoeLayerProgram>(layer.feed_forward.storage())) {
            return true;
        }
    }
    return false;
}

void CompiledModelProgram::validate() const {
    if (hidden <= 0 || !std::isfinite(logits_multiplier) ||
        !std::isfinite(logits_divisor) || logits_divisor <= 0.0f ||
        !std::isfinite(final_logit_softcap) || final_logit_softcap < 0.0f) {
        throw std::invalid_argument("compiled model has invalid residual/output policy");
    }
    per_layer_input.validate();
    for (size_t index = 0; index < norm_after_layers.size(); ++index) {
        const int layer = norm_after_layers[index];
        if (layer < 0 || layer >= static_cast<int>(layers.size()) - 1 ||
            (index > 0 && norm_after_layers[index - 1] >= layer)) {
            throw std::invalid_argument("invalid compiled norm boundary");
        }
    }
    for (const auto& layer : layers) {
        switch (layer.chunk_capability) {
        case CompiledChunkCapability::Native:
        case CompiledChunkCapability::SequentialAdapter:
        case CompiledChunkCapability::Unsupported:
            break;
        default:
            throw std::invalid_argument("invalid compiled chunk capability");
        }
        if (layer.weight_request_indices.empty()) {
            throw std::invalid_argument("compiled layer has no weight plan");
        }
        for (const std::size_t index : layer.weight_request_indices) {
            if (index >= weight_request_count) {
                throw std::invalid_argument("compiled layer has invalid weight index");
            }
        }
        if (const auto* state = layer.state_layout()) state->validate();
        layer.feed_forward.validate();
        if (!std::isfinite(layer.residual.multiplier)) {
            throw std::invalid_argument("compiled layer has invalid residual multiplier");
        }
    }
}

} // namespace celeg