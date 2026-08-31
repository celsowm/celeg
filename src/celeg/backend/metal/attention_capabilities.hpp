#pragma once

#include "celeg/backend/attention_capabilities.hpp"

#include <stdexcept>

namespace celeg {

inline constexpr AttentionBackendCapabilities metal_attention_capabilities() {
    return {
        .full_causal = true,
        .sliding_window = true,
        .bidirectional = false,
        .prefix_lm = false,
        .block_sparse = false,
        .dynamic_sparse = false,
        .external_memory = false,
        .alibi = false,
        .relative_position_bias = false,
        .no_position = false,
        .rope = true,
        .multi_axis_rope = false,
        .standard_execution = true,
        .latent_execution = false,
        .factorized_latent_execution = false,
    };
}

inline void validate_metal_attention_capabilities(
    const CompiledModelProgram& program) {
    validate_attention_backend_capabilities(
        program, "Metal", metal_attention_capabilities());

    for (const CompiledLayerProgram& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;
        if (const auto* sliding =
                std::get_if<SlidingWindowPattern>(&attention.pattern);
            sliding && sliding->window <= 0) {
            throw std::invalid_argument(
                "Metal sliding-window attention requires a positive window");
        }
        if (!std::holds_alternative<PrivateKv>(attention.kv_sharing)) {
            throw std::invalid_argument(
                "Metal attention currently supports private KV only");
        }
        if (!std::holds_alternative<OrdinaryKvStateSpec>(attention.state)) {
            throw std::invalid_argument(
                "Metal attention currently supports ordinary KV state only");
        }
        const auto& ordinary = std::get<OrdinaryKvStateSpec>(attention.state);
        if (ordinary.storage.key != StateScalarType::BF16 ||
            ordinary.storage.value != StateScalarType::BF16) {
            throw std::invalid_argument(
                "Metal attention currently supports BF16 KV state only");
        }
        if (attention.output_gate.has_value()) {
            throw std::invalid_argument(
                "Metal attention currently does not support output gates");
        }
        if (!std::holds_alternative<NoAttentionOutputTransformSpec>(
                attention.output_transform)) {
            throw std::invalid_argument(
                "Metal attention currently does not support output transforms");
        }
    }
}

}
