#pragma once

#include "celeg/backend/attention_capabilities.hpp"

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
}

}
