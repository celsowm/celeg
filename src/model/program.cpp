#include "celeg/model/program.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

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

} // namespace

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
        throw std::invalid_argument("per-layer input width exceeds CUDA integer range");
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
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("per-layer input row product exceeds CUDA integer range");
    }
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
            layer.mixer_kind() == MixerKind::Attention
                ? CompiledMixer::Attention : CompiledMixer::ShortConvolution,
            layer.feed_forward_kind() == FeedForwardKind::Dense
                ? CompiledFeedForward::Dense : CompiledFeedForward::MixtureOfExperts,
            {}};
        for (std::size_t request_index = 0;
             request_index < model.weight_plan.requests.size(); ++request_index) {
            if (model.weight_plan.requests[request_index].layer == static_cast<int>(layer_index)) {
                compiled.weight_request_indices.push_back(request_index);
            }
        }
        if (compiled.weight_request_indices.empty()) {
            throw std::invalid_argument("layer has no resolved weight requests");
        }
        program.layers.push_back(std::move(compiled));
    }
    program.validate();
    return program;
}

} // namespace celeg
