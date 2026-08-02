#include "naming_policy.hpp"

#include <stdexcept>

namespace celeg {

std::vector<std::string> Gemma4TensorNamingPolicy::candidates(
    const TensorRequest& request) const {
    const std::string prefix = "model.language_model.";
    const std::string layer = request.layer < 0 ? std::string{} :
        "layers." + std::to_string(request.layer) + ".";
    switch (request.role) {
    case TensorRole::TokenEmbedding:
        return {prefix + "embed_tokens.weight"};
    case TensorRole::LanguageModelHead:
        return {prefix + "embed_tokens.weight"};
    case TensorRole::FinalNorm:
        return {prefix + "norm.weight"};
    case TensorRole::AttentionInputNorm:
        return {prefix + layer + "input_layernorm.weight"};
    case TensorRole::AttentionQuery:
        return {prefix + layer + "self_attn.q_proj.weight"};
    case TensorRole::AttentionQueryNorm:
        return {prefix + layer + "self_attn.q_norm.weight"};
    case TensorRole::AttentionKey:
        return {prefix + layer + "self_attn.k_proj.weight"};
    case TensorRole::AttentionKeyNorm:
        return {prefix + layer + "self_attn.k_norm.weight"};
    case TensorRole::AttentionValue:
        return {prefix + layer + "self_attn.v_proj.weight"};
    case TensorRole::AttentionValueNorm:
        return {};
    case TensorRole::AttentionOutput:
        return {prefix + layer + "self_attn.o_proj.weight"};
    case TensorRole::AttentionPostNorm:
        return {prefix + layer + "post_attention_layernorm.weight"};
    case TensorRole::FfnInputNorm:
        return {prefix + layer + "pre_feedforward_layernorm.weight"};
    case TensorRole::FfnGate:
        return {prefix + layer + "mlp.gate_proj.weight"};
    case TensorRole::FfnUp:
        return {prefix + layer + "mlp.up_proj.weight"};
    case TensorRole::FfnDown:
        return {prefix + layer + "mlp.down_proj.weight"};
    case TensorRole::FfnOutputNorm:
        return {prefix + layer + "post_feedforward_layernorm.weight"};
    case TensorRole::PerLayerEmbedding:
        return {prefix + "embed_tokens_per_layer.weight"};
    case TensorRole::PerLayerContextProjection:
        return {prefix + "per_layer_model_projection.weight"};
    case TensorRole::PerLayerProjection:
        return {prefix + layer + "per_layer_projection.weight"};
    case TensorRole::PerLayerProjectionNorm:
        return {prefix + "per_layer_projection_norm.weight"};
    case TensorRole::PerLayerInputGate:
        return {prefix + layer + "per_layer_input_gate.weight"};
    case TensorRole::PerLayerInputNorm:
        return {prefix + layer + "post_per_layer_input_norm.weight"};
    case TensorRole::LayerScalar:
        return {prefix + layer + "layer_scalar"};
    default:
        return {};
    }
}

} // namespace celeg
