#include "naming_policy.hpp"

namespace celeg {
namespace {
std::vector<std::string> names(std::string suffix, int layer) {
    const std::string a = "backbone.layers." + std::to_string(layer) + "." + suffix;
    const std::string b = "model.backbone.layers." + std::to_string(layer) + "." + suffix;
    return {a, b};
}
}

std::vector<std::string> NemotronHTensorNamingPolicy::candidates(
    const TensorRequest& request) const {
    if (request.layer < 0) {
        switch (request.role) {
        case TensorRole::TokenEmbedding:
            return {"backbone.embeddings.weight", "model.backbone.embeddings.weight"};
        case TensorRole::LanguageModelHead:
            return {"lm_head.weight", "model.lm_head.weight"};
        case TensorRole::FinalNorm:
            return {"backbone.norm_f.weight", "model.backbone.norm_f.weight",
                    "backbone.norm.weight", "model.backbone.norm.weight"};
        default: return {};
        }
    }
    switch (request.role) {
    case TensorRole::AttentionInputNorm: return names("norm.weight", request.layer);
    case TensorRole::AttentionQuery: return names("mixer.q_proj.weight", request.layer);
    case TensorRole::AttentionKey: return names("mixer.k_proj.weight", request.layer);
    case TensorRole::AttentionValue: return names("mixer.v_proj.weight", request.layer);
    case TensorRole::AttentionOutput: return names("mixer.o_proj.weight", request.layer);
    case TensorRole::Mamba2Input: return names("mixer.in_proj.weight", request.layer);
    case TensorRole::Mamba2Conv: return names("mixer.conv1d.weight", request.layer);
    case TensorRole::Mamba2ConvBias: return names("mixer.conv1d.bias", request.layer);
    case TensorRole::Mamba2DtBias: return names("mixer.dt_bias", request.layer);
    case TensorRole::Mamba2ALog: return names("mixer.A_log", request.layer);
    case TensorRole::Mamba2D: return names("mixer.D", request.layer);
    case TensorRole::Mamba2Norm: return names("mixer.norm.weight", request.layer);
    case TensorRole::Mamba2Output: return names("mixer.out_proj.weight", request.layer);
    case TensorRole::FfnUp: return names("mixer.up_proj.weight", request.layer);
    case TensorRole::FfnDown: return names("mixer.down_proj.weight", request.layer);
    default: return {};
    }
}

} // namespace celeg
