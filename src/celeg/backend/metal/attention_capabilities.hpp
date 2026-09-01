#pragma once

#include "celeg/backend/attention_capabilities.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

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
        .alibi = true,
        .relative_position_bias = true,
        .no_position = true,
        .rope = true,
        .multi_axis_rope = true,
        .standard_execution = true,
        .latent_execution = false,
        .factorized_latent_execution = false,
    };
}

inline void validate_metal_attention_capabilities(
    const CompiledModelProgram& program) {
    validate_attention_backend_capabilities(
        program, "Metal", metal_attention_capabilities());

    std::unordered_map<int, const AttentionSpec*> shared_owner;
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
        if (const auto* publisher =
                std::get_if<SharedKvPublisher>(&attention.kv_sharing)) {
            if (publisher->group < 0 || shared_owner.contains(publisher->group)) {
                throw std::invalid_argument(
                    "Metal shared KV requires one non-negative publisher per group");
            }
            shared_owner.emplace(publisher->group, &attention);
        } else if (const auto* consumer =
                       std::get_if<SharedKvConsumer>(&attention.kv_sharing)) {
            const auto owner = shared_owner.find(consumer->group);
            if (consumer->group < 0 || owner == shared_owner.end()) {
                throw std::invalid_argument(
                    "Metal shared KV consumer requires an earlier publisher");
            }
            if (owner->second->key_value_heads != attention.key_value_heads ||
                owner->second->head_dim != attention.head_dim) {
                throw std::invalid_argument(
                    "Metal shared KV consumer geometry does not match its publisher");
            }
            if (!std::holds_alternative<NoAttentionOutputTransformSpec>(
                    attention.output_transform)) {
                throw std::invalid_argument(
                    "Metal shared KV consumers do not yet support current-value output transforms");
            }
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
            if (std::abs(rope->rotary_fraction - 1.0) > 1.0e-12) {
                throw std::invalid_argument(
                    "Metal attention currently requires full-width RoPE");
            }
            if (!std::holds_alternative<NoRopeScaling>(rope->scaling)) {
                throw std::invalid_argument(
                    "Metal attention currently does not support RoPE scaling");
            }
        }
        if (const MultiAxisRopeSpec* multi = attention.multi_axis_position()) {
            if (!multi->interleaved || multi->axes != 3 ||
                multi->base.pairing != RopePairingKind::SplitHalf) {
                throw std::invalid_argument(
                    "Metal M-RoPE requires three interleaved axes with split-half pairing");
            }
            if (std::abs(multi->base.rotary_fraction - 1.0) > 1.0e-12 ||
                !std::holds_alternative<NoRopeScaling>(multi->base.scaling)) {
                throw std::invalid_argument(
                    "Metal M-RoPE currently requires full-width unscaled RoPE");
            }
            if (std::abs(multi->base.theta - 10000.0) > 1.0e-9) {
                throw std::invalid_argument(
                    "Metal M-RoPE currently requires theta 10000");
            }
            const int pairs = attention.head_dim / 2;
            if (multi->sections[0] + multi->sections[1] + multi->sections[2] != pairs) {
                throw std::invalid_argument(
                    "Metal M-RoPE sections do not match the rotary dimension");
            }
        }
        if (attention.output_gate.has_value() &&
            attention.output_gate->packed_with_query &&
            attention.output_gate->granularity == AttentionGateGranularity::HeadWise) {
            throw std::invalid_argument(
                "Metal packed attention gates currently require element-wise semantics");
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
