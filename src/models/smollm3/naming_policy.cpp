#include "naming_policy.hpp"

#include <stdexcept>

namespace celeg::detail {
namespace {

std::string layer_name(int layer, std::string_view suffix) {
    if (layer < 0) throw std::invalid_argument("tensor role requires a layer");
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

} // namespace

std::vector<std::string> SmolLm3TensorNamingPolicy::candidates(
    const TensorRequest& request) const {
    switch (request.role) {
    case TensorRole::TokenEmbedding:
        return {"model.embed_tokens.weight"};
    case TensorRole::LanguageModelHead:
        return {"lm_head.weight"};
    case TensorRole::FinalNorm:
        return {"model.norm.weight"};
    case TensorRole::AttentionInputNorm:
        return {layer_name(request.layer, "input_layernorm.weight")};
    case TensorRole::AttentionQuery:
        return {layer_name(request.layer, "self_attn.q_proj.weight")};
    case TensorRole::AttentionKey:
        return {layer_name(request.layer, "self_attn.k_proj.weight")};
    case TensorRole::AttentionValue:
        return {layer_name(request.layer, "self_attn.v_proj.weight")};
    case TensorRole::AttentionOutput:
        return {layer_name(request.layer, "self_attn.o_proj.weight")};
    case TensorRole::FfnInputNorm:
        return {layer_name(request.layer, "post_attention_layernorm.weight")};
    case TensorRole::FfnGate:
        return {layer_name(request.layer, "mlp.gate_proj.weight")};
    case TensorRole::FfnUp:
        return {layer_name(request.layer, "mlp.up_proj.weight")};
    case TensorRole::FfnDown:
        return {layer_name(request.layer, "mlp.down_proj.weight")};
    default:
        return {};
    }
}

} // namespace celeg::detail
