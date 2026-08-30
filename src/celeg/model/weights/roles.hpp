#pragma once

#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/norm.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

enum class TensorRole : uint8_t {
    TokenEmbedding,
    LanguageModelHead,
    FinalNorm,
    AttentionInputNorm,
    AttentionQuery,
    AttentionQueryNorm,
    AttentionKey,
    AttentionKeyNorm,
    AttentionValue,
    AttentionGate,
    AttentionRelativePositionBias,
    AttentionLatentQuery,
    AttentionLatentQueryRope,
    AttentionLatentKey,
    AttentionLatentValue,
    AttentionLatentKeyRope,
    AttentionLatentOutput,
    AttentionLatentQueryProjection,
    AttentionLatentQueryExpansion,
    AttentionLatentQueryNorm,
    AttentionLatentKeyProjection,
    AttentionLatentKeyNorm,
    AttentionLatentExpansion,
    AttentionOutput,
    AttentionValueNorm,
    AttentionPostNorm,
    FfnInputNorm,
    FfnOutputNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    ShortConvInput,
    ShortConvKernel,
    ShortConvOutput,
    GatedDeltaNetQkv,
    GatedDeltaNetZ,
    GatedDeltaNetAlpha,
    GatedDeltaNetBeta,
    GatedDeltaNetDtBias,
    GatedDeltaNetALog,
    GatedDeltaNetConv,
    GatedDeltaNetNorm,
    GatedDeltaNetOutput,
    GatedDeltaNetQuery,
    GatedDeltaNetKey,
    GatedDeltaNetValue,
    GatedDeltaNetDecay,
    GatedDeltaNetOutputGate,
    GatedDeltaNetQueryConv,
    GatedDeltaNetKeyConv,
    GatedDeltaNetValueConv,
    Mamba2Input,
    Mamba2Conv,
    Mamba2ConvBias,
    Mamba2DtBias,
    Mamba2ALog,
    Mamba2D,
    Mamba2Norm,
    Mamba2Output,
    PerLayerEmbedding,
    PerLayerContextProjection,
    PerLayerProjection,
    PerLayerProjectionNorm,
    PerLayerInputGate,
    PerLayerInputNorm,
    LayerScalar,
    MoeRouter,
    MoeRouterBias,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
    MoePackedGateUp,
    MoePackedDown,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    MoeSharedGateWeight,
};

std::string_view tensor_role_name(TensorRole role);

struct TensorRequest {
    TensorRole role;
    int layer = -1;
    int expert = -1;
    std::vector<int64_t> expected_shape;
    std::optional<std::string> source_name;
    int physical_layer = -1;
    NormWeightKind norm_weight_kind = NormWeightKind::Scale;
};

struct ResolvedTensor {
    std::string source_name;
    HostTensorView view;
};

/**
 * Resolved source names for one MoE layer's routed-expert tensors, extracted
 * from the authoritative weight plan. Exactly one family is populated:
 * either per-expert individual names (one gate/up/down entry per expert) or
 * the packed tensor pair. Backends consume this value instead of re-deriving
 * names or packed-vs-individual layout from checkpoint spellings.
 */
struct MoeExpertTensorNames {
    std::vector<std::string> gate;
    std::vector<std::string> up;
    std::vector<std::string> down;
    std::string packed_gate_up;
    std::string packed_down;

    bool packed() const { return !packed_gate_up.empty(); }
};

class ITensorNamingPolicy {
public:
    virtual ~ITensorNamingPolicy() = default;
    virtual std::vector<std::string> candidates(const TensorRequest& request) const = 0;
};


class TensorResolver {
public:
    TensorResolver(const IWeightRepository& repository,
                   const ITensorNamingPolicy& naming_policy)
        : repository_(repository), naming_policy_(naming_policy) {}

    ResolvedTensor resolve(const TensorRequest& request) const;

private:
    const IWeightRepository& repository_;
    const ITensorNamingPolicy& naming_policy_;
};

std::string resolved_tensor_name(std::span<const TensorRequest> requests,
                                 TensorRole role, int layer = -1,
                                 int expert = -1);

MoeExpertTensorNames moe_expert_tensor_names(
    std::span<const TensorRequest> requests, int layer, int num_experts);

}
