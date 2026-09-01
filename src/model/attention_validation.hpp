#pragma once

#include "celeg/model/graph.hpp"

#include <stdexcept>

namespace celeg {

inline void validate_attention_representation(const AttentionSpec& attention) {
    if (!attention.output_gate.has_value()) return;

    const SigmoidAttentionGateSpec& gate = *attention.output_gate;
    if (gate.packed_with_query &&
        gate.granularity == AttentionGateGranularity::HeadWise) {
        throw std::invalid_argument(
            "packed HeadWise attention output gates have no canonical representation");
    }
    if (gate.packed_with_query && attention.uses_latent_state()) {
        throw std::invalid_argument(
            "packed attention output gates require ordinary query projection");
    }
}

}
