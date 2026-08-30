#pragma once

#include "celeg/runtime/cache/expert_source.hpp"

#include <span>

namespace celeg {

struct SharedModelWeights;

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

}
