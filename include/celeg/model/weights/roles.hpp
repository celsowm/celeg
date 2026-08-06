#pragma once

#include "celeg/checkpoint/weight_repository.hpp"

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
    PerLayerEmbedding,
    PerLayerContextProjection,
    PerLayerProjection,
    PerLayerProjectionNorm,
    PerLayerInputGate,
    PerLayerInputNorm,
    LayerScalar,
    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
};

std::string_view tensor_role_name(TensorRole role);

struct TensorRequest {
    TensorRole role;
    int layer = -1;
    int expert = -1;
    std::vector<int64_t> expected_shape;
    std::optional<std::string> source_name;
};

struct ResolvedTensor {
    std::string source_name;
    HostTensorView view;
};

class ITensorNamingPolicy {
public:
    virtual ~ITensorNamingPolicy() = default;
    virtual std::vector<std::string> candidates(const TensorRequest& request) const = 0;
};

// Concrete naming policies (CelegTensorNamingPolicy, GraniteTensorNamingPolicy,
// Gemma4TensorNamingPolicy, ...) live in their owning src/models/<arch>/
// module. This generic header depends on ITensorNamingPolicy only, so adding
// an architecture never requires touching a central tensor-name switch.

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

} // namespace celeg
