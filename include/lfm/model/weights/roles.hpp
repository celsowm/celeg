#pragma once

#include "lfm/checkpoint/formats/safetensors.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lfm {

enum class TensorRole : uint8_t {
    TokenEmbedding,
    LanguageModelHead,
    FinalNorm,
    AttentionInputNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    FfnInputNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    ShortConvInput,
    ShortConvKernel,
    ShortConvOutput,
    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown,
};

struct TensorRequest {
    TensorRole role;
    int layer = -1;
    int expert = -1;
    std::vector<int64_t> expected_shape;
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

// Canonical LFM names are kept in this policy rather than in loading or CUDA
// code. Alternative checkpoint spellings can be added as another policy.
class LfmTensorNamingPolicy final : public ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(const TensorRequest& request) const override;
};

class GraniteTensorNamingPolicy final : public ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(const TensorRequest& request) const override;
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

} // namespace lfm
