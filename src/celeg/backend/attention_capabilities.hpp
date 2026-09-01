#pragma once

#include "celeg/model/program.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

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

inline bool shared_attention_state_compatible(const AttentionSpec& publisher,
                                              const AttentionSpec& consumer) {
    if (publisher.state.index() != consumer.state.index() ||
        publisher.key_value_heads != consumer.key_value_heads ||
        publisher.head_dim != consumer.head_dim ||
        publisher.state_storage_granularity() !=
            consumer.state_storage_granularity() ||
        publisher.state_storage_paged() != consumer.state_storage_paged()) {
        return false;
    }

    if (const auto* publisher_ordinary =
            std::get_if<OrdinaryKvStateSpec>(&publisher.state)) {
        const auto& consumer_ordinary =
            std::get<OrdinaryKvStateSpec>(consumer.state);
        return publisher_ordinary->storage.key == consumer_ordinary.storage.key &&
               publisher_ordinary->storage.value == consumer_ordinary.storage.value;
    }

    const auto& publisher_latent =
        std::get<LatentAttentionStateSpec>(publisher.state);
    const auto& consumer_latent =
        std::get<LatentAttentionStateSpec>(consumer.state);
    return publisher_latent.latent_rank == consumer_latent.latent_rank &&
           publisher_latent.rope_head_dim == consumer_latent.rope_head_dim &&
           publisher_latent.nope_head_dim == consumer_latent.nope_head_dim &&
           publisher_latent.decoupled_rope == consumer_latent.decoupled_rope &&
           publisher_latent.storage.latent == consumer_latent.storage.latent &&
           publisher_latent.storage.rotary == consumer_latent.storage.rotary;
}

inline void validate_shared_attention_contracts(
    const CompiledModelProgram& program) {
    std::unordered_map<int, const AttentionSpec*> publishers;
    for (const auto& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;

        if (const auto* publisher =
                std::get_if<SharedKvPublisher>(&attention.kv_sharing)) {
            if (publisher->group < 0 || publishers.contains(publisher->group)) {
                throw std::invalid_argument(
                    "shared KV requires one non-negative publisher per group");
            }
            publishers.emplace(publisher->group, &attention);
            continue;
        }

        const auto* consumer =
            std::get_if<SharedKvConsumer>(&attention.kv_sharing);
        if (!consumer) continue;
        const auto owner = publishers.find(consumer->group);
        if (consumer->group < 0 || owner == publishers.end()) {
            throw std::invalid_argument(
                "shared KV consumer requires an earlier publisher");
        }
        if (!shared_attention_state_compatible(*owner->second, attention)) {
            throw std::invalid_argument(
                "shared KV consumer state does not match its publisher");
        }
    }
}

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
