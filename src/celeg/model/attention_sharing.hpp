#pragma once

#include "celeg/model/program.hpp"

#include <stdexcept>
#include <unordered_map>

namespace celeg {

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

}
