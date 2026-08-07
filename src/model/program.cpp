#include "celeg/model/program.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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
         group_count * experts_per_group != expert_count)) {
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
    append_field(out, has_expert_bias);
    append_field(out, routed_scaling);
    return fingerprint_text(out.str());
}

void ExpertMlpProgram::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("invalid compiled MoE expert MLP program");
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

PerLayerInputPlan PerLayerInputPlan::derive(const ResolvedModel& model) {
    PerLayerInputPlan result;
    const RuntimeTopology& topology = model.topology;
    if (!topology.has_per_layer_input) {
        return result;
    }
    if (topology.num_hidden_layers <= 0 || topology.per_layer_input_size <= 0 ||
        topology.hidden <= 0) {
        throw std::invalid_argument("invalid per-layer input dimensions");
    }
    result.enabled = true;
    result.layer_count = topology.num_hidden_layers;
    result.input_size = topology.per_layer_input_size;
    result.packed_width = checked_product(
        static_cast<std::size_t>(result.layer_count),
        static_cast<std::size_t>(result.input_size),
        "per-layer input width overflows size_t");
    if (result.packed_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "per-layer input width exceeds the compiled execution index range");
    }
    result.token_scale = std::sqrt(static_cast<float>(result.input_size));
    result.context_scale = 1.0f / std::sqrt(static_cast<float>(topology.hidden));
    result.residual_scale = kPerLayerResidualScale;
    result.norm_epsilon = topology.numerical_policy.norm_eps;

    if (model.graph.layers.empty()) {
        throw std::invalid_argument("per-layer input requires resolved layer specifications");
    }
    result.activation = model.graph.layers.front().per_layer_input.activation;
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
    const std::size_t elements = checked_product(
        rows, packed_width, "per-layer input row product overflows size_t");
    return elements;
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

bool CompiledModelProgram::has_moe() const {
    for (const auto& layer : layers) {
        if (layer.feed_forward == CompiledFeedForward::MixtureOfExperts) return true;
    }
    return false;
}

void CompiledModelProgram::validate() const {
    per_layer_input.validate();
    for (const auto& layer : layers) {
        if (layer.weight_request_indices.empty()) {
            throw std::invalid_argument("compiled layer has no weight plan");
        }
        for (const std::size_t index : layer.weight_request_indices) {
            if (index >= weight_request_count) {
                throw std::invalid_argument("compiled layer has invalid weight index");
            }
        }
        const bool has_mixer_spec =
            layer.attention.has_value() || layer.short_convolution.has_value() ||
            layer.gated_delta_net.has_value() || layer.mamba2.has_value();
        if (layer.mixer == CompiledMixer::MlpOnly) {
            if (has_mixer_spec) {
                throw std::invalid_argument("MLP-only layer has mixer semantics");
            }
        } else if (!has_mixer_spec) {
            throw std::invalid_argument("compiled mixer has no semantic specification");
        }
        if (layer.mixer == CompiledMixer::Attention && !layer.attention) {
            throw std::invalid_argument("compiled attention layer has no attention semantics");
        }
        if (layer.mixer == CompiledMixer::ShortConvolution && !layer.short_convolution) {
            throw std::invalid_argument("compiled convolution layer has no convolution semantics");
        }
        if (layer.mixer == CompiledMixer::GatedDeltaNet && !layer.gated_delta_net) {
            throw std::invalid_argument("compiled GatedDeltaNet layer has no semantics");
        }
        if (layer.mixer == CompiledMixer::Mamba2 && !layer.mamba2) {
            throw std::invalid_argument("compiled Mamba2 layer has no semantics");
        }
        if (layer.feed_forward_intermediate <= 0) {
            throw std::invalid_argument("compiled layer has no FFN width");
        }
        if (layer.feed_forward == CompiledFeedForward::MixtureOfExperts) {
            if (!layer.moe) {
                throw std::invalid_argument("compiled MoE layer has no semantics");
            }
            layer.moe->validate();
        } else if (layer.moe) {
            throw std::invalid_argument("dense layer has an MoE semantic program");
        }
    }
}

void validate_moe_backend_capabilities(const CompiledModelProgram& program,
                                       std::string_view backend,
                                       MoeBackendCapabilities capabilities) {
    for (const auto& layer : program.layers) {
        if (!layer.moe) continue;
        const MoeLayerProgram& moe = *layer.moe;
        if (moe.router.selection == MoeSelectionKind::GroupedTopK &&
            !capabilities.grouped_selection) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support grouped MoE selection");
        }
        if (moe.shared && !capabilities.shared_experts) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support shared MoE experts");
        }
        if (moe.routed.payload.layout == MoePayloadLayout::Stacked &&
            !capabilities.stacked_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support stacked MoE payloads");
        }
        if (moe.routed.payload.layout == MoePayloadLayout::Fused &&
            !capabilities.fused_payload) {
            throw std::invalid_argument(std::string(backend) +
                " backend does not support fused MoE payloads");
        }
    }
}

CompiledModelProgram build_model_program(const ResolvedModel& model) {
    if (model.graph.layers.empty()) throw std::invalid_argument("model has no layers");
    CompiledModelProgram program;
    program.identity = model.provenance.identity;
    program.per_layer_input = PerLayerInputPlan::derive(model);
    program.layers.reserve(model.graph.layers.size());

    program.weight_request_count = model.weight_plan.requests.size();
    for (std::size_t request_index = 0;
         request_index < model.weight_plan.requests.size(); ++request_index) {
        if (model.weight_plan.requests[request_index].layer < 0) {
            program.unlayered_weight_request_indices.push_back(request_index);
        }
    }

    for (std::size_t layer_index = 0; layer_index < model.graph.layers.size(); ++layer_index) {
        const LayerSpec& layer = model.graph.layers[layer_index];
        CompiledLayerProgram compiled{
            layer.mixer_kind() == MixerKind::Attention ? CompiledMixer::Attention :
            layer.mixer_kind() == MixerKind::ShortConvolution ? CompiledMixer::ShortConvolution :
            layer.mixer_kind() == MixerKind::GatedDeltaNet ? CompiledMixer::GatedDeltaNet :
            layer.mixer_kind() == MixerKind::Mamba2 ? CompiledMixer::Mamba2 :
            CompiledMixer::MlpOnly,
            layer.feed_forward_kind() == FeedForwardKind::Dense
                ? CompiledFeedForward::Dense : CompiledFeedForward::MixtureOfExperts,
            layer.execute_feed_forward,
            {}, {}, {}, {},
            0, ActivationKind::SwiGLU,
            {}, std::nullopt};
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
                compiled.attention = mixer;
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                compiled.short_convolution = mixer;
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                compiled.gated_delta_net = mixer;
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                compiled.mamba2 = mixer;
            }
        }, layer.mixer);
        std::visit([&](const auto& feed_forward) {
            using FeedForward = std::decay_t<decltype(feed_forward)>;
            if constexpr (std::is_same_v<FeedForward, DenseFeedForwardSpec>) {
                compiled.feed_forward_intermediate = feed_forward.intermediate_size;
                compiled.feed_forward_activation = feed_forward.activation;
            } else if constexpr (std::is_same_v<FeedForward, MlpBlockSpec>) {
                compiled.feed_forward_intermediate = feed_forward.intermediate_size;
                compiled.feed_forward_activation = feed_forward.activation;
            } else if constexpr (std::is_same_v<FeedForward, MixtureOfExpertsSpec>) {
                compiled.feed_forward_intermediate = feed_forward.intermediate_size;
            }
        }, layer.feed_forward);
        for (std::size_t request_index = 0;
             request_index < model.weight_plan.requests.size(); ++request_index) {
            if (model.weight_plan.requests[request_index].layer == static_cast<int>(layer_index)) {
                compiled.weight_request_indices.push_back(request_index);
            }
        }
        if (compiled.weight_request_indices.empty()) {
            throw std::invalid_argument("layer has no resolved weight requests");
        }
        if (const auto* moe = std::get_if<MixtureOfExpertsSpec>(&layer.feed_forward)) {
            MoeLayerProgram semantic;
            semantic.router.expert_count = moe->num_experts;
            semantic.router.experts_per_token = moe->experts_per_token;
            semantic.router.normalization = moe->normalize_topk
                ? MoeNormalizationKind::SumSelected : MoeNormalizationKind::None;
            semantic.router.score = moe->router_softmax
                ? MoeRouterScoreKind::SoftmaxLogits
                : MoeRouterScoreKind::SigmoidProbabilities;
            semantic.router.has_expert_bias = moe->use_expert_bias;
            semantic.router.routed_scaling = moe->routed_scaling_factor;
            semantic.router.selection = moe->routing_group_count > 0
                ? MoeSelectionKind::GroupedTopK : MoeSelectionKind::TopK;
            semantic.router.group_count = moe->routing_group_count;
            semantic.router.experts_per_group = moe->routing_experts_per_group;
            semantic.routed.mlp.hidden_size = model.topology.hidden;
            semantic.routed.mlp.intermediate_size = moe->intermediate_size;
            const std::size_t expert_matrix_elements =
                static_cast<std::size_t>(model.topology.hidden) *
                static_cast<std::size_t>(moe->intermediate_size);
            semantic.routed.payload.regions = {
                {TensorRole::MoeExpertGate, expert_matrix_elements},
                {TensorRole::MoeExpertUp, expert_matrix_elements},
                {TensorRole::MoeExpertDown, expert_matrix_elements}};
            semantic.output.has_shared_expert = moe->has_shared_expert;
            semantic.output.combine_order = moe->shared_before_routed
                ? MoeCombineOrder::SharedThenRouted : MoeCombineOrder::RoutedThenShared;
            if (moe->has_shared_expert) {
                semantic.shared = SharedExpertProgram{
                    ExpertMlpProgram{MoeActivation::SwiGLU, model.topology.hidden,
                                     moe->shared_intermediate_size > 0
                                         ? moe->shared_intermediate_size
                                         : moe->intermediate_size}};
            }
            semantic.residency.expert_count = moe->num_experts;
            semantic.residency.payload_elements = 3 * expert_matrix_elements;
            compiled.moe = std::move(semantic);
        }
        program.layers.push_back(std::move(compiled));
    }
    std::ostringstream semantic;
    for (const auto& layer : program.layers) {
        semantic << (layer.moe ? layer.moe->fingerprint() : "dense") << ';';
    }
    program.semantic_fingerprint = fingerprint_text(semantic.str());
    program.validate();
    return program;
}

} // namespace celeg
