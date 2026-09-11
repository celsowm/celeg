#pragma once

#include "celeg/backend/attention_capabilities.hpp"

#include <cmath>
#include <stdexcept>

namespace celeg {

inline constexpr AttentionBackendCapabilities metal_attention_capabilities() {
    return {
        .full_causal = true,
        .sliding_window = true,
        .bidirectional = true,
        .prefix_lm = true,
        .block_sparse = false,
        .dynamic_sparse = false,
        .external_memory = false,
        .alibi = true,
        .relative_position_bias = true,
        .no_position = true,
        .rope = true,
        .multi_axis_rope = true,
        .standard_execution = true,
        .latent_execution = false,
        .factorized_latent_execution = false,
        .value_norm = true,
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
        if (attention.value_norm) attention.value_norm->validate();
        if (const auto* sliding =
                std::get_if<SlidingWindowPattern>(&attention.pattern);
            sliding && sliding->window <= 0) {
            throw std::invalid_argument(
                "Metal sliding-window attention requires a positive window");
        }
        if (const auto* prefix = std::get_if<PrefixLmPattern>(&attention.pattern);
            prefix && prefix->prefix_length <= 0) {
            throw std::invalid_argument(
                "Metal Prefix-LM attention requires a positive prefix length");
        }
        const bool dense_noncausal =
            std::holds_alternative<BidirectionalPattern>(attention.pattern) ||
            std::holds_alternative<PrefixLmPattern>(attention.pattern);
        if (dense_noncausal &&
            !std::holds_alternative<NoAttentionBiasSpec>(attention.bias)) {
            throw std::invalid_argument(
                "Metal bidirectional/Prefix-LM attention currently requires no attention bias");
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
        if (const RopePositionSpec* rope = attention.rope_position()) {
            if (!(rope->rotary_fraction > 0.0) ||
                rope->rotary_fraction > 1.0 ||
                !std::isfinite(rope->rotary_fraction)) {
                throw std::invalid_argument(
                    "Metal RoPE rotary fraction must be finite and in (0, 1]");
            }
            const int rotary_dimension =
                rope->resolved_rotary_dimension(attention.head_dim);
            if (rotary_dimension <= 0 || (rotary_dimension % 2) != 0) {
                throw std::invalid_argument(
                    "Metal RoPE rotary dimension must be positive and even");
            }
        }
        if (const MultiAxisRopeSpec* multi = attention.multi_axis_position()) {
            if (multi->axes != 3 ||
                multi->base.pairing != RopePairingKind::SplitHalf) {
                throw std::invalid_argument(
                    "Metal M-RoPE requires three axes with split-half pairing");
            }
            if (std::abs(multi->base.rotary_fraction - 1.0) > 1.0e-12 ||
                !std::holds_alternative<NoRopeScaling>(multi->base.scaling)) {
                throw std::invalid_argument(
                    "Metal M-RoPE currently requires full-width unscaled RoPE");
            }
            if (!(multi->base.theta > 0.0) || !std::isfinite(multi->base.theta)) {
                throw std::invalid_argument(
                    "Metal M-RoPE theta must be finite and positive");
            }
            const int pairs = attention.head_dim / 2;
            if (multi->sections[0] + multi->sections[1] + multi->sections[2] != pairs) {
                throw std::invalid_argument(
                    "Metal M-RoPE sections do not match the rotary dimension");
            }
        }
        if (const auto* transform =
                std::get_if<OrthogonalizeCurrentValueSpec>(
                    &attention.output_transform)) {
            if (!(transform->minimum_norm_squared > 0.0f) ||
                !std::isfinite(transform->minimum_norm_squared)) {
                throw std::invalid_argument(
                    "Metal attention output transform requires a positive finite norm floor");
            }
        }
    }
}

}