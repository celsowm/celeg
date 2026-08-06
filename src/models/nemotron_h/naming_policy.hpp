#pragma once

#include "celeg/model/weights/roles.hpp"

namespace celeg {

class NemotronHTensorNamingPolicy final : public ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(const TensorRequest& request) const override;
};

} // namespace celeg
