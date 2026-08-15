#include "celeg/model/inference.hpp"

#include "support.hpp"

#include <unordered_map>
#include <unordered_set>

namespace celeg {

const TensorRoleBinding* TensorRoleBindings::find(TensorRole role, int layer,
                                                  int expert) const noexcept {
    for (const auto& binding : values) {
        if (binding.role == role && binding.layer == layer && binding.expert == expert) {
            return &binding;
        }
    }
    return nullptr;
}

void TensorRoleBindings::validate() const {
    std::unordered_set<std::string> identities;
    std::unordered_map<std::string, TensorRole> tensor_roles;
    for (const auto& binding : values) {
        if (binding.source_name.empty() || binding.shape.empty()) {
            inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                                   "tensor role binding is incomplete");
        }
        const std::string identity = std::to_string(static_cast<int>(binding.role)) + ":" +
            std::to_string(binding.layer) + ":" + std::to_string(binding.expert);
        if (!identities.insert(identity).second) {
            inference_detail::fail(
                ResolutionFailureKind::AmbiguousTensorBinding,
                "tensor role is bound more than once: " + identity);
        }
        const auto [tensor_it, tensor_inserted] =
            tensor_roles.emplace(binding.source_name, binding.role);
        if (!tensor_inserted) {
            const bool tied_embedding_pair = binding.layer == -1 && binding.expert == -1 &&
                ((tensor_it->second == TensorRole::TokenEmbedding &&
                  binding.role == TensorRole::LanguageModelHead) ||
                 (tensor_it->second == TensorRole::LanguageModelHead &&
                  binding.role == TensorRole::TokenEmbedding));
            if (!tied_embedding_pair) {
                inference_detail::fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "one source tensor is assigned to more than one semantic role: " +
                        binding.source_name);
            }
        }
    }
}

TensorRoleBindings BindingSolver::solve(
    const std::vector<TensorRoleBinding>& candidates) const {
    TensorRoleBindings result;
    std::unordered_map<std::string, size_t> assignment;
    for (const auto& candidate : candidates) {
        const std::string identity = std::to_string(static_cast<int>(candidate.role)) + ":" +
            std::to_string(candidate.layer) + ":" + std::to_string(candidate.expert);
        const auto [it, inserted] = assignment.emplace(identity, result.values.size());
        if (!inserted) {
            inference_detail::fail(
                ResolutionFailureKind::AmbiguousTensorBinding,
                "more than one complete tensor assignment remains for " + identity);
        }
        (void)it;
        result.values.push_back(candidate);
    }
    result.validate();
    return result;
}

}
