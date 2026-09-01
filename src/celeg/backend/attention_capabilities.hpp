#pragma once

#include "celeg/model/attention_sharing.hpp"
#include "celeg/model/program.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace celeg {

struct AttentionBackendCapabilities {
    bool full_causal = false;
    bool sliding_window = false;
    bool bidirectional = false;
    bool prefix_lm = false;
    bool block_sparse = false;
    bool dynamic_sparse = false;
    bool external_memory = false;
    bool alibi = false;
    bool relative_position_bias = false;
    bool no_position = false;
    bool rope = false;
    bool multi_axis_rope = false;
    bool standard_execution = false;
    bool latent_execution = false;
    bool factorized_latent_execution = false;
};

inline void validate_attention_backend_capabilities(
    const CompiledModelProgram& program,
    std::string_view backend,
    AttentionBackendCapabilities capabilities) {
    validate_shared_attention_contracts(program);

    const auto unsupported = [&](std::string_view feature) {
        return std::invalid_argument(std::string(backend) +
            " backend does not support " + std::string(feature));
    };

    for (const auto& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;

        if (attention.uses_external_memory() && !capabilities.external_memory) {
            throw unsupported("external-memory attention");
        }
        if (std::holds_alternative<FullCausalPattern>(attention.pattern) &&
            !capabilities.full_causal) throw unsupported("full-causal attention");
        if (std::holds_alternative<SlidingWindowPattern>(attention.pattern) &&
            !capabilities.sliding_window) throw unsupported("sliding-window attention");
        if (std::holds_alternative<BidirectionalPattern>(attention.pattern) &&
            !capabilities.bidirectional) throw unsupported("bidirectional attention");
        if (std::holds_alternative<PrefixLmPattern>(attention.pattern) &&
            !capabilities.prefix_lm) throw unsupported("prefix-LM attention");
        if (std::holds_alternative<BlockSparsePattern>(attention.pattern) &&
            !capabilities.block_sparse) throw unsupported("block-sparse attention");
        if (std::holds_alternative<DynamicSparsePattern>(attention.pattern) &&
            !capabilities.dynamic_sparse) throw unsupported("dynamic-sparse attention");
        if (std::holds_alternative<AlibiBiasSpec>(attention.bias) &&
            !capabilities.alibi) throw unsupported("ALiBi attention bias");
        if (std::holds_alternative<RelativePositionBiasSpec>(attention.bias) &&
            !capabilities.relative_position_bias) {
            throw unsupported("relative-position attention bias");
        }
        if (std::holds_alternative<NoPositionEncodingSpec>(attention.position) &&
            !capabilities.no_position) throw unsupported("attention without position encoding");
        if (std::holds_alternative<RopePositionSpec>(attention.position) &&
            !capabilities.rope) throw unsupported("RoPE attention position encoding");
        if (std::holds_alternative<MultiAxisRopeSpec>(attention.position) &&
            !capabilities.multi_axis_rope) throw unsupported("multi-axis RoPE attention position encoding");

        switch (compiled->execution.kind) {
        case AttentionExecutionKind::Standard:
            if (!capabilities.standard_execution) {
                throw unsupported("standard attention execution");
            }
            break;
        case AttentionExecutionKind::Latent:
            if (!capabilities.latent_execution) {
                throw unsupported("latent attention execution");
            }
            break;
        case AttentionExecutionKind::FactorizedLatent:
            if (!capabilities.factorized_latent_execution) {
                throw unsupported("factorized latent attention execution");
            }
            break;
        }
    }
}

}
