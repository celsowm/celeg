#include "naming_policy.hpp"

#include <stdexcept>

namespace celeg::detail {
namespace {

std::string layer_name(const TensorRequest& request, std::string_view suffix) {
    const int layer = request.physical_layer >= 0 ? request.physical_layer : request.layer;
    if (layer < 0) throw std::invalid_argument("Nanbeige tensor role requires a layer");
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

} // namespace

std::vector<std::string> Nanbeige42TensorNamingPolicy::candidates(
    const TensorRequest& request) const {
    switch (request.role) {
    case TensorRole::TokenEmbedding: return {"model.embed_tokens.weight"};
    case TensorRole::LanguageModelHead: return {"lm_head.weight"};
    case TensorRole::FinalNorm: return {"model.norm.weight"};
    case TensorRole::AttentionInputNorm:
    case TensorRole::FfnInputNorm:
        return {layer_name(request, request.role == TensorRole::AttentionInputNorm
            ? "input_layernorm.weight" : "post_attention_layernorm.weight")};
    case TensorRole::AttentionQuery: return {layer_name(request, "self_attn.q_proj.weight")};
    case TensorRole::AttentionKey: return {layer_name(request, "self_attn.k_proj.weight")};
    case TensorRole::AttentionValue: return {layer_name(request, "self_attn.v_proj.weight")};
    case TensorRole::AttentionOutput: return {layer_name(request, "self_attn.o_proj.weight")};
    case TensorRole::FfnGate: return {layer_name(request, "mlp.gate_proj.weight")};
    case TensorRole::FfnUp: return {layer_name(request, "mlp.up_proj.weight")};
    case TensorRole::FfnDown: return {layer_name(request, "mlp.down_proj.weight")};
    default: return {};
    }
}

} // namespace celeg::detail
