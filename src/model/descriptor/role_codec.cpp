#include "detail.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace celeg::descriptor_detail {

TensorRole parse_role(std::string_view name) {
    static const std::unordered_map<std::string_view, TensorRole> roles = {
        {"TokenEmbedding", TensorRole::TokenEmbedding},
        {"LanguageModelHead", TensorRole::LanguageModelHead},
        {"FinalNorm", TensorRole::FinalNorm},
        {"AttentionInputNorm", TensorRole::AttentionInputNorm},
        {"AttentionQuery", TensorRole::AttentionQuery},
        {"AttentionQueryNorm", TensorRole::AttentionQueryNorm},
        {"AttentionKey", TensorRole::AttentionKey},
        {"AttentionKeyNorm", TensorRole::AttentionKeyNorm},
        {"AttentionValue", TensorRole::AttentionValue},
        {"AttentionGate", TensorRole::AttentionGate},
        {"AttentionRelativePositionBias", TensorRole::AttentionRelativePositionBias},
        {"AttentionLatentQuery", TensorRole::AttentionLatentQuery},
        {"AttentionLatentQueryRope", TensorRole::AttentionLatentQueryRope},
        {"AttentionLatentKey", TensorRole::AttentionLatentKey},
        {"AttentionLatentValue", TensorRole::AttentionLatentValue},
        {"AttentionLatentKeyRope", TensorRole::AttentionLatentKeyRope},
        {"AttentionLatentOutput", TensorRole::AttentionLatentOutput},
        {"AttentionLatentQueryProjection", TensorRole::AttentionLatentQueryProjection},
        {"AttentionLatentQueryExpansion", TensorRole::AttentionLatentQueryExpansion},
        {"AttentionLatentQueryNorm", TensorRole::AttentionLatentQueryNorm},
        {"AttentionLatentKeyProjection", TensorRole::AttentionLatentKeyProjection},
        {"AttentionLatentKeyNorm", TensorRole::AttentionLatentKeyNorm},
        {"AttentionLatentExpansion", TensorRole::AttentionLatentExpansion},
        {"AttentionOutput", TensorRole::AttentionOutput},
        {"AttentionPostNorm", TensorRole::AttentionPostNorm},
        {"FfnInputNorm", TensorRole::FfnInputNorm},
        {"FfnOutputNorm", TensorRole::FfnOutputNorm},
        {"FfnGate", TensorRole::FfnGate},
        {"FfnUp", TensorRole::FfnUp},
        {"FfnDown", TensorRole::FfnDown},
        {"PerLayerEmbedding", TensorRole::PerLayerEmbedding},
        {"PerLayerContextProjection", TensorRole::PerLayerContextProjection},
        {"PerLayerProjection", TensorRole::PerLayerProjection},
        {"PerLayerProjectionNorm", TensorRole::PerLayerProjectionNorm},
        {"PerLayerInputGate", TensorRole::PerLayerInputGate},
        {"PerLayerInputNorm", TensorRole::PerLayerInputNorm},
        {"LayerScalar", TensorRole::LayerScalar},
        {"ShortConvInput", TensorRole::ShortConvInput},
        {"ShortConvKernel", TensorRole::ShortConvKernel},
        {"ShortConvOutput", TensorRole::ShortConvOutput},
        {"GatedDeltaNetQkv", TensorRole::GatedDeltaNetQkv},
        {"GatedDeltaNetZ", TensorRole::GatedDeltaNetZ},
        {"GatedDeltaNetAlpha", TensorRole::GatedDeltaNetAlpha},
        {"GatedDeltaNetBeta", TensorRole::GatedDeltaNetBeta},
        {"GatedDeltaNetDtBias", TensorRole::GatedDeltaNetDtBias},
        {"GatedDeltaNetALog", TensorRole::GatedDeltaNetALog},
        {"GatedDeltaNetConv", TensorRole::GatedDeltaNetConv},
        {"GatedDeltaNetNorm", TensorRole::GatedDeltaNetNorm},
        {"GatedDeltaNetOutput", TensorRole::GatedDeltaNetOutput},
        {"GatedDeltaNetQuery", TensorRole::GatedDeltaNetQuery},
        {"GatedDeltaNetKey", TensorRole::GatedDeltaNetKey},
        {"GatedDeltaNetValue", TensorRole::GatedDeltaNetValue},
        {"GatedDeltaNetDecay", TensorRole::GatedDeltaNetDecay},
        {"GatedDeltaNetOutputGate", TensorRole::GatedDeltaNetOutputGate},
        {"GatedDeltaNetQueryConv", TensorRole::GatedDeltaNetQueryConv},
        {"GatedDeltaNetKeyConv", TensorRole::GatedDeltaNetKeyConv},
        {"GatedDeltaNetValueConv", TensorRole::GatedDeltaNetValueConv},
        {"Mamba2Input", TensorRole::Mamba2Input},
        {"Mamba2Conv", TensorRole::Mamba2Conv},
        {"Mamba2ConvBias", TensorRole::Mamba2ConvBias},
        {"Mamba2DtBias", TensorRole::Mamba2DtBias},
        {"Mamba2ALog", TensorRole::Mamba2ALog},
        {"Mamba2D", TensorRole::Mamba2D},
        {"Mamba2Norm", TensorRole::Mamba2Norm},
        {"Mamba2Output", TensorRole::Mamba2Output},
        {"MoeRouter", TensorRole::MoeRouter},
        {"MoeRouterBias", TensorRole::MoeRouterBias},
        {"MoeExpertGate", TensorRole::MoeExpertGate},
        {"MoeExpertUp", TensorRole::MoeExpertUp},
        {"MoeExpertDown", TensorRole::MoeExpertDown},
        {"MoePackedGateUp", TensorRole::MoePackedGateUp},
        {"MoePackedDown", TensorRole::MoePackedDown},
        {"MoeSharedGate", TensorRole::MoeSharedGate},
        {"MoeSharedUp", TensorRole::MoeSharedUp},
        {"MoeSharedDown", TensorRole::MoeSharedDown},
        {"MoeSharedGateWeight", TensorRole::MoeSharedGateWeight},
    };
    const auto it = roles.find(name);
    if (it == roles.end()) {
        throw std::invalid_argument("descriptor has unsupported tensor role: " +
                                    std::string(name));
    }
    return it->second;
}

} // namespace celeg::descriptor_detail
