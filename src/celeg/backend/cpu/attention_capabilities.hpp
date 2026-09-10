#pragma once

#include "celeg/backend/attention_capabilities.hpp"

namespace celeg {

inline constexpr AttentionBackendCapabilities cpu_attention_capabilities() {
    return {
        .full_causal = true,
        .sliding_window = true,
        .bidirectional = true,
        .prefix_lm = true,
        .block_sparse = true,
        .dynamic_sparse = false,
        .external_memory = true,
        .alibi = true,
        .relative_position_bias = true,
        .no_position = true,
        .rope = true,
        .multi_axis_rope = true,
        .standard_execution = true,
        .latent_execution = true,
        .factorized_latent_execution = true,
        .value_norm = true,
    };
}

}
