#include "celeg/model/weights/roles.hpp"

#include <sstream>
#include <stdexcept>
#include <iterator>

namespace celeg {

std::string_view tensor_role_name(TensorRole role) {
    static constexpr std::string_view names[] = {
        "token_embedding", "language_model_head", "final_norm",
        "attention_input_norm", "attention_query", "attention_query_norm",
        "attention_key", "attention_key_norm", "attention_value",
        "attention_output", "attention_value_norm", "attention_post_norm",
        "ffn_input_norm", "ffn_output_norm", "ffn_gate", "ffn_up", "ffn_down",
        "short_conv_input", "short_conv_kernel", "short_conv_output",
        "per_layer_embedding", "per_layer_context_projection", "per_layer_projection",
        "per_layer_projection_norm", "per_layer_input_gate", "per_layer_input_norm",
        "layer_scalar", "moe_router", "moe_expert_gate", "moe_expert_up", "moe_expert_down"
    };
    const auto index = static_cast<size_t>(role);
    return index < std::size(names) ? names[index] : "unknown";
}

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
    message << "required tensor role is missing: " << tensor_role_name(request.role)
            << " layer=" << request.layer << " expert=" << request.expert
            << " candidates=[";
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (i != 0) message << ", ";
        message << candidates[i];
    }
    message << "]";
    throw std::runtime_error(message.str());
}

} // namespace celeg
