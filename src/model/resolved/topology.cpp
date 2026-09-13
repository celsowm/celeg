#include "celeg/model/resolved.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace celeg {

ExecutionTopology ExecutionTopology::derive(const ModelGraph& graph) {
    ExecutionTopology result;
    const std::size_t layer_count = graph.layers.size();
    if (layer_count == 0) {
        throw std::invalid_argument("cannot derive runtime shape from an empty graph");
    }
    result.num_hidden_layers = static_cast<int>(layer_count);
    result.attention_slot_for_layer.assign(layer_count, -1);
    result.layer_for_attention_slot.clear();
    result.attention_layer_count = 0;
    result.conv_layer_count = 0;
    result.gated_delta_net_layer_count = 0;
    result.mamba2_layer_count = 0;
    result.mlp_only_layer_count = 0;
    for (std::size_t index = 0; index < layer_count; ++index) {
        const LayerSpec& layer = graph.layers[index];
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, AttentionSpec>) {
                result.attention_slot_for_layer[index] = result.attention_layer_count++;
                result.layer_for_attention_slot.push_back(static_cast<int>(index));
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                ++result.conv_layer_count;
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                ++result.gated_delta_net_layer_count;
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                ++result.mamba2_layer_count;
            } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                ++result.mlp_only_layer_count;
            } else {
                static_assert(always_false_v<Mixer>, "unhandled mixer derivation variant");
            }
        }, layer.mixer);
    }
    return result;
}

RuntimeTopology compose_runtime_topology(CheckpointDimensions checkpoint,
                                         const ModelGraph& graph) {
    RuntimeTopology topology;
    topology.dims = std::move(checkpoint);
    topology.exec = ExecutionTopology::derive(graph);
    return topology;
}

std::string ExecutionTopology::summary() const {
    std::ostringstream out;
    out << "layers=" << num_hidden_layers
        << " attention_layers=" << attention_layer_count
        << " conv_layers=" << conv_layer_count
        << " gated_delta_layers=" << gated_delta_net_layer_count
        << " mamba2_layers=" << mamba2_layer_count
        << " mlp_only_layers=" << mlp_only_layer_count;
    return out.str();
}

void ExecutionTopology::validate() const {
    if (num_hidden_layers <= 0) {
        throw std::runtime_error("invalid resolved model topology");
    }
    if (attention_layer_count + conv_layer_count + gated_delta_net_layer_count +
        mamba2_layer_count + mlp_only_layer_count != num_hidden_layers) {
        throw std::runtime_error("resolved layer counts are inconsistent");
    }
}

std::string RuntimeTopology::summary() const {
    std::ostringstream out;
    out << exec.summary()
        << " mtp_layers=" << dims.mtp_num_hidden_layers
        << " vocab=" << dims.vocab_size;
    return out.str();
}

void RuntimeTopology::validate() const {
    dims.validate();
    exec.validate();
    if (!dims.checkpoint_layer_for_layer.empty() &&
        static_cast<int>(dims.checkpoint_layer_for_layer.size()) !=
            exec.num_hidden_layers) {
        throw std::runtime_error("checkpoint layer mapping length mismatch");
    }
}

}
