#include "celeg/model/program.hpp"

#include <stdexcept>

namespace celeg {

bool CompiledModelProgram::has_moe() const {
    for (const auto& layer : layers) {
        if (layer.feed_forward == CompiledFeedForward::MixtureOfExperts) return true;
    }
    return false;
}

void CompiledModelProgram::validate() const {
    if (architecture_id.empty()) throw std::invalid_argument("compiled program has no architecture");
    if (source_format.empty()) throw std::invalid_argument("compiled program has no checkpoint format");
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
    program.identity = model.identity;
    program.architecture_id = model.architecture_id;
    program.source_format = model.definition.source_format;
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
