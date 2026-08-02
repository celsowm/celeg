#pragma once

#include "celeg/model/weights/roles.hpp"

namespace celeg {

// Canonical LFM/Celeg tensor names are kept in this policy rather than in
// loading or CUDA code. Alternative checkpoint spellings can be added as
// another policy.
class CelegTensorNamingPolicy final : public ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(const TensorRequest& request) const override;
};

} // namespace celeg
