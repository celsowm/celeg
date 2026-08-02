#include "celeg/model/weights/roles.hpp"

#include <sstream>
#include <stdexcept>

namespace celeg {

ResolvedTensor TensorResolver::resolve(const TensorRequest& request) const {
    const std::vector<std::string> candidates = naming_policy_.candidates(request);
    for (const std::string& candidate : candidates) {
        if (!repository_.contains(candidate)) continue;
        ResolvedTensor resolved{candidate, repository_.tensor(candidate)};
        if (!request.expected_shape.empty() &&
            resolved.view.shape != request.expected_shape) {
            std::ostringstream message;
            message << "tensor shape mismatch for " << candidate;
            throw std::runtime_error(message.str());
        }
        return resolved;
    }
    std::ostringstream message;
    message << "required tensor role is missing: "
            << static_cast<int>(request.role);
    throw std::runtime_error(message.str());
}

} // namespace celeg
