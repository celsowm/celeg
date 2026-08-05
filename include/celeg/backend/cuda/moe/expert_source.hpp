#pragma once

#include "celeg/runtime/cache/expert_source.hpp"

#include <span>

namespace celeg {

struct SharedModelWeights;

// CUDA's source adapter owns the mapping from the compiled expert payload to
// the available sidecar or random-access repository. The cache only supplies
// a byte span and never learns how that span is populated.
class CudaExpertSource final : public IExpertSource {
public:
    explicit CudaExpertSource(const SharedModelWeights& weights)
        : weights_(&weights) {}

    void read(const ExpertPayloadRequest& request,
              std::span<std::byte> destination) const override;
    void read(int layer, int expert, std::span<std::byte> destination) const;

private:
    const SharedModelWeights* weights_ = nullptr;
};

} // namespace celeg
