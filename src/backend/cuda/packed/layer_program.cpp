#include "celeg/backend/cuda/packed/layer_program.hpp"

#include <stdexcept>
#include <type_traits>

namespace celeg {

PackedLayerProgram PackedLayerProgram::compile(const CompiledModelProgram& program) {
    if (program.layers.empty()) {
        throw std::invalid_argument("packed layer program has incomplete mixer topology");
    }
    std::vector<PackedLayerBinding> layers;
    layers.reserve(program.layers.size());
    for (const CompiledLayerProgram& layer : program.layers) {
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, CompiledAttentionProgram>) {
                layers.push_back({PackedLayerKind::Attention});
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
                layers.push_back({PackedLayerKind::ShortConvolution});
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                layers.push_back({PackedLayerKind::GatedDeltaNet});
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                layers.push_back({PackedLayerKind::Mamba2});
            } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                layers.push_back({PackedLayerKind::MlpOnly});
            } else {
                static_assert(always_false_v<Mixer>, "unsupported compiled mixer variant");
            }
        }, layer.mixer);
    }
    return PackedLayerProgram(std::move(layers));
}

} // namespace celeg
