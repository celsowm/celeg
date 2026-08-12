#include "celeg/backend/cuda/packed_layer_program.hpp"

#include <stdexcept>

namespace celeg {

PackedLayerProgram PackedLayerProgram::compile(const ExecutionTopology& shape) {
    if (shape.num_hidden_layers <= 0 ||
        shape.mixer_kinds.size() != static_cast<size_t>(shape.num_hidden_layers)) {
        throw std::invalid_argument("packed layer program has incomplete mixer topology");
    }
    std::vector<PackedLayerBinding> layers;
    layers.reserve(shape.mixer_kinds.size());
    for (const MixerKind kind : shape.mixer_kinds) {
        switch (kind) {
        case MixerKind::Attention:
            layers.push_back({PackedLayerKind::Attention});
            break;
        case MixerKind::ShortConvolution:
            layers.push_back({PackedLayerKind::ShortConvolution});
            break;
        case MixerKind::GatedDeltaNet:
            layers.push_back({PackedLayerKind::GatedDeltaNet});
            break;
        case MixerKind::Mamba2:
            layers.push_back({PackedLayerKind::Mamba2});
            break;
        case MixerKind::MlpOnly:
            layers.push_back({PackedLayerKind::MlpOnly});
            break;
        default:
            throw std::invalid_argument("packed layer program has unsupported mixer kind");
        }
    }
    return PackedLayerProgram(std::move(layers));
}

} // namespace celeg
