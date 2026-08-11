#include "celeg/model/weights/roles.hpp"

#include <sstream>
#include <stdexcept>
#include <iterator>

namespace celeg {

std::string_view tensor_role_name(TensorRole role) {
    static constexpr std::string_view names[] = {
        "token_embedding", "language_model_head", "final_norm",
        "attention_input_norm", "attention_query", "attention_query_norm",
        "attention_key", "attention_key_norm", "attention_value", "attention_gate",
        "attention_relative_position_bias",
        "attention_latent_query", "attention_latent_query_rope",
        "attention_latent_key", "attention_latent_value",
        "attention_latent_key_rope", "attention_latent_output",
        "attention_latent_query_projection", "attention_latent_query_expansion",
        "attention_latent_query_norm", "attention_latent_key_projection",
        "attention_latent_key_norm", "attention_latent_expansion",
        "attention_output", "attention_value_norm", "attention_post_norm",
        "ffn_input_norm", "ffn_output_norm", "ffn_gate", "ffn_up", "ffn_down",
        "short_conv_input", "short_conv_kernel", "short_conv_output",
        "gated_delta_net_qkv", "gated_delta_net_z", "gated_delta_net_alpha",
        "gated_delta_net_beta", "gated_delta_net_dt_bias", "gated_delta_net_a_log",
        "gated_delta_net_conv", "gated_delta_net_norm", "gated_delta_net_output",
        "gated_delta_net_query", "gated_delta_net_key", "gated_delta_net_value",
        "gated_delta_net_decay", "gated_delta_net_output_gate",
        "gated_delta_net_query_conv", "gated_delta_net_key_conv",
        "gated_delta_net_value_conv",
        "mamba2_input", "mamba2_conv", "mamba2_conv_bias", "mamba2_dt_bias",
        "mamba2_a_log", "mamba2_d", "mamba2_norm", "mamba2_output",
        "per_layer_embedding", "per_layer_context_projection", "per_layer_projection",
        "per_layer_projection_norm", "per_layer_input_gate", "per_layer_input_norm",
        "layer_scalar", "moe_router", "moe_router_bias", "moe_expert_gate", "moe_expert_up", "moe_expert_down",
        "moe_packed_gate_up", "moe_packed_down", "moe_shared_gate",
        "moe_shared_up", "moe_shared_down", "moe_shared_gate_weight"
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

std::string resolved_tensor_name(std::span<const TensorRequest> requests,
                                 TensorRole role, int layer, int expert) {
    for (const TensorRequest& request : requests) {
        if (request.role != role || request.layer != layer || request.expert != expert) {
            continue;
        }
        if (request.source_name) return *request.source_name;
        throw std::logic_error("tensor request has no resolved source name: " +
                               std::string(tensor_role_name(role)));
    }
    throw std::out_of_range("resolved tensor request not found: " +
                            std::string(tensor_role_name(role)));
}

} // namespace celeg
