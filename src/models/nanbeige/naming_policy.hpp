#pragma once

#include "celeg/model/weights/roles.hpp"

namespace celeg::detail {

class Nanbeige42TensorNamingPolicy final : public ITensorNamingPolicy {
public:
    std::vector<std::string> candidates(const TensorRequest& request) const override;
};

} // namespace celeg::detail
