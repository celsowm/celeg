#pragma once

#include "celeg/model/program.hpp"

#include <string_view>

namespace celeg {

/**
 * @brief Variant descriptor for attention encode.
 *
 * Captures the branching that previously duplicated across
 * `encode_attention` and `encode_attention_batch`. The resolver is
 * pure CPU and unit-testable.
 */
struct AttentionVariant {
    bool owns_kv = false;
    bool fused_per_head = false;
    bool qk_publishes_kv = false;
    bool tiled_candidate = false;
    bool no_position = false;
    bool split_half_rope = false;
    bool multi_axis = false;
    std::string_view qk_kernel{};
    std::string_view attention_kernel{};
};

/// @brief Resolve the attention variant for the given program and geometry.
AttentionVariant resolve_attention_variant(
    const CompiledAttentionProgram& attention,
    uint32_t rows,
    uint32_t base_position,
    uint32_t head_dim,
    bool owns_kv) noexcept;

/// @brief Select the attention kernel name for decode.
std::string_view select_decode_attention_kernel(
    bool has_relative,
    bool has_alibi,
    uint32_t window_size) noexcept;

/// @brief Select the attention kernel name for batch.
std::string_view select_batch_attention_kernel(
    bool has_relative,
    bool has_alibi,
    uint32_t window_size) noexcept;

}
