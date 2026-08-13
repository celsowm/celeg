#include "celeg/backend/cuda/packed/layer_program.hpp"

#include <stdexcept>

namespace celeg {

PackedLayerProgram PackedLayerProgram::compile(const CompiledModelProgram& program) {
    if (program.layers.empty()) {
        throw std::invalid_argument("packed layer program has incomplete mixer topology");
    }
    std::vector<PackedLayerBinding> layers;
    layers.reserve(program.layers.size());
    for (const CompiledLayerProgram& layer : program.layers) {
        const CompiledMixer kind = layer.mixer;
        switch (kind) {
        case CompiledMixer::Attention:
            layers.push_back({PackedLayerKind::Attention});
            break;
        case CompiledMixer::ShortConvolution:
            layers.push_back({PackedLayerKind::ShortConvolution});
            break;
        case CompiledMixer::GatedDeltaNet:
            layers.push_back({PackedLayerKind::GatedDeltaNet});
            break;
        case CompiledMixer::Mamba2:
            layers.push_back({PackedLayerKind::Mamba2});
            break;
        case CompiledMixer::MlpOnly:
            layers.push_back({PackedLayerKind::MlpOnly});
            break;
        default:
            throw std::invalid_argument("packed layer program has unsupported mixer kind");
        }
    }
    return PackedLayerProgram(std::move(layers));
}

} // namespace celeg
