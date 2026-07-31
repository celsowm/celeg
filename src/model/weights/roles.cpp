#include "celeg/model/weights/roles.hpp"

#include <sstream>
#include <stdexcept>

namespace celeg {
namespace {

std::string layer_name(int layer, std::string_view suffix) {
    if (layer < 0) throw std::invalid_argument("tensor role requires a layer");
    return "model.layers." + std::to_string(layer) + "." + std::string(suffix);
}

std::string role_suffix(TensorRole role) {
    switch (role) {
    case TensorRole::AttentionInputNorm: return "operator_norm.weight";
    case TensorRole::AttentionQuery: return "self_attn.q_proj.weight";
    case TensorRole::AttentionKey: return "self_attn.k_proj.weight";
    case TensorRole::AttentionValue: return "self_attn.v_proj.weight";
    case TensorRole::AttentionOutput: return "self_attn.out_proj.weight";
    case TensorRole::FfnInputNorm: return "ffn_norm.weight";
    case TensorRole::FfnGate: return "feed_forward.w1.weight";
    case TensorRole::FfnUp: return "feed_forward.w3.weight";
    case TensorRole::FfnDown: return "feed_forward.w2.weight";
    case TensorRole::ShortConvInput: return "conv.input_proj.weight";
    case TensorRole::ShortConvKernel: return "conv.conv.weight";
    case TensorRole::ShortConvOutput: return "conv.output_proj.weight";
    case TensorRole::MoeRouter: return "feed_forward.gate.weight";
    default: return {};
    }
}

}

std::vector<std::string> CelegTensorNamingPolicy::candidates(
    const TensorRequest& request) const {
    switch (request.role) {
    case TensorRole::TokenEmbedding:
        return {"model.embed_tokens.weight", "model.embedding.weight"};
    case TensorRole::LanguageModelHead:
        return {"lm_head.weight", "model.lm_head.weight"};
    case TensorRole::FinalNorm:
        return {"model.embedding_norm.weight", "model.norm.weight",
                "model.final_layernorm.weight"};
    case TensorRole::MoeExpertGate:
    case TensorRole::MoeExpertUp:
    case TensorRole::MoeExpertDown: {
        if (request.layer < 0 || request.expert < 0) {
            throw std::invalid_argument("expert tensor role requires layer and expert");
        }
        const char* suffix = request.role == TensorRole::MoeExpertGate ? "w1" :
            (request.role == TensorRole::MoeExpertUp ? "w3" : "w2");
        return {"model.layers." + std::to_string(request.layer) +
                ".feed_forward.experts." + std::to_string(request.expert) +
                "." + suffix + ".weight"};
    }
    default: {
        const std::string suffix = role_suffix(request.role);
        if (suffix.empty()) throw std::invalid_argument("tensor role has no LFM spelling");
        return {layer_name(request.layer, suffix)};
    }
    }
}

std::vector<std::string> GraniteTensorNamingPolicy::candidates(
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
        throw std::invalid_argument("tensor role has no Granite spelling");
    }
}

ResolvedTensor TensorResolver::resolve(const TensorRequest& request) const {
    const std::vector<std::string> candidates = naming_policy_.candidates(request);
    for (const std::string& candidate : candidates) {
        if (!repository_.contains(candidate)) continue;
        ResolvedTensor resolved{candidate, repository_.tensor(candidate)};
        if (!request.expected_shape.empty() &&
            resolved.view.shape != request.expected_shape) {
            std::ostringstream message;
            message << "tensor shape mismatch for " << candidate;
            throw std::runtime_error(message.str());
        }
        return resolved;
    }
    std::ostringstream message;
    message << "required tensor role is missing: "
            << static_cast<int>(request.role);
    throw std::runtime_error(message.str());
}

} // namespace celeg
