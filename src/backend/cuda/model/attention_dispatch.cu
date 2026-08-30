#include "detail/compiled_model.hpp"

#include <stdexcept>

namespace celeg {

void CudaCompiledModel::enqueue_decode_attention(
    Layer& layer, LayerCommon& common_layer, int layer_index) {
    const CompiledLayerProgram& semantics = resources_.program_.layers.at(
        static_cast<std::size_t>(layer_index));
    const auto* attention = std::get_if<CompiledAttentionProgram>(&semantics.mixer);
    if (!attention) {
        throw std::logic_error("CUDA attention has no compiled attention program");
    }
    switch (attention->execution.kind) {
    case AttentionExecutionKind::Standard:
        enqueue_decode_standard_attention(layer, common_layer, layer_index);
        return;
    case AttentionExecutionKind::Latent:
        enqueue_decode_latent_attention(layer, common_layer, layer_index);
        return;
    case AttentionExecutionKind::FactorizedLatent:
        enqueue_decode_factorized_latent_attention(layer, common_layer, layer_index);
        return;
    }
    throw std::logic_error("unknown compiled CUDA attention execution kind");
}

}
