#include "naming_policy.hpp"

namespace celeg {

std::vector<std::string> Qwen35TensorNamingPolicy::candidates(const TensorRequest& request) const {
    const std::string prefix = "model.language_model.";
    const std::string layer = request.layer < 0 ? std::string{} :
        "layers." + std::to_string(request.layer) + ".";
    const std::string linear = prefix + layer + "linear_attn.";
    switch (request.role) {
    case TensorRole::TokenEmbedding:
    case TensorRole::LanguageModelHead: return {prefix + "embed_tokens.weight"};
    case TensorRole::FinalNorm: return {prefix + "norm.weight"};
    case TensorRole::AttentionInputNorm: return {prefix + layer + "input_layernorm.weight"};
    case TensorRole::AttentionPostNorm: return {prefix + layer + "post_attention_layernorm.weight"};
    case TensorRole::AttentionQuery: return {prefix + layer + "self_attn.q_proj.weight"};
    case TensorRole::AttentionQueryNorm: return {prefix + layer + "self_attn.q_norm.weight"};
    case TensorRole::AttentionKey: return {prefix + layer + "self_attn.k_proj.weight"};
    case TensorRole::AttentionKeyNorm: return {prefix + layer + "self_attn.k_norm.weight"};
    case TensorRole::AttentionValue: return {prefix + layer + "self_attn.v_proj.weight"};
    case TensorRole::AttentionOutput: return {prefix + layer + "self_attn.o_proj.weight"};
    case TensorRole::FfnInputNorm: return {};
    case TensorRole::FfnGate: return {prefix + layer + "mlp.gate_proj.weight"};
    case TensorRole::FfnUp: return {prefix + layer + "mlp.up_proj.weight"};
    case TensorRole::FfnDown: return {prefix + layer + "mlp.down_proj.weight"};
    case TensorRole::GatedDeltaNetQkv: return {linear + "in_proj_qkv.weight"};
    case TensorRole::GatedDeltaNetZ: return {linear + "in_proj_z.weight"};
    case TensorRole::GatedDeltaNetAlpha: return {linear + "in_proj_a.weight"};
    case TensorRole::GatedDeltaNetBeta: return {linear + "in_proj_b.weight"};
    case TensorRole::GatedDeltaNetDtBias: return {linear + "dt_bias"};
    case TensorRole::GatedDeltaNetALog: return {linear + "A_log"};
    case TensorRole::GatedDeltaNetConv: return {linear + "conv1d.weight"};
    case TensorRole::GatedDeltaNetNorm: return {linear + "norm.weight"};
    case TensorRole::GatedDeltaNetOutput: return {linear + "out_proj.weight"};
    default: return {};
    }
}

} // namespace celeg
