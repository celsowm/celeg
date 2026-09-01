#pragma once

#include "celeg/model/program.hpp"

#include <stdexcept>

namespace celeg {

inline bool cuda_constrained_standard_pattern(const AttentionSpec& attention) {
    return std::holds_alternative<BidirectionalPattern>(attention.pattern) ||
           std::holds_alternative<PrefixLmPattern>(attention.pattern) ||
           std::holds_alternative<BlockSparsePattern>(attention.pattern) ||
           std::holds_alternative<DynamicSparsePattern>(attention.pattern);
}

inline bool cuda_constrained_pattern_bias_supported(
    const AttentionSpec& attention) {
    if (std::holds_alternative<NoAttentionBiasSpec>(attention.bias)) return true;
    return std::holds_alternative<BidirectionalPattern>(attention.pattern) &&
           std::holds_alternative<RelativePositionBiasSpec>(attention.bias);
}

inline void validate_cuda_attention_semantics(
    const CompiledAttentionProgram& compiled) {
    const AttentionSpec& attention = compiled.semantics;

    if (const auto* prefix_lm = std::get_if<PrefixLmPattern>(&attention.pattern);
        prefix_lm && prefix_lm->prefix_length <= 0) {
        throw std::invalid_argument(
            "CUDA prefix-LM attention requires a positive prefix length");
    }

    if (const auto* block_sparse = std::get_if<BlockSparsePattern>(&attention.pattern);
        block_sparse &&
        (block_sparse->block_size <= 0 || block_sparse->local_blocks <= 0 ||
         block_sparse->global_blocks < 0)) {
        throw std::invalid_argument(
            "CUDA block-sparse attention requires positive block/local sizes and non-negative global blocks");
    }

    if (const auto* dynamic_sparse = std::get_if<DynamicSparsePattern>(&attention.pattern);
        dynamic_sparse &&
        (dynamic_sparse->block_size <= 0 ||
         dynamic_sparse->max_selected_blocks <= 0 ||
         dynamic_sparse->max_selected_blocks > 32)) {
        throw std::invalid_argument(
            "CUDA dynamic-sparse attention requires a positive block size and 1..32 selected blocks");
    }

    if (cuda_constrained_standard_pattern(attention)) {
        if (!cuda_constrained_pattern_bias_supported(attention)) {
            throw std::invalid_argument(
                "CUDA constrained attention pattern does not support this attention bias");
        }
        if (compiled.execution.kind != AttentionExecutionKind::Standard) {
            throw std::invalid_argument(
                "CUDA constrained attention patterns currently support standard attention only");
        }
    }

    if (const auto* relative =
            std::get_if<RelativePositionBiasSpec>(&attention.bias)) {
        if (relative->bidirectional &&
            !std::holds_alternative<BidirectionalPattern>(attention.pattern)) {
            throw std::invalid_argument(
                "CUDA bidirectional relative-position bias requires bidirectional attention");
        }
        if (compiled.execution.kind != AttentionExecutionKind::Standard) {
            throw std::invalid_argument(
                "CUDA relative-position bias currently supports standard attention only");
        }
    }

    if (compiled.execution.kind != AttentionExecutionKind::Standard) {
        const auto* latent = attention.latent_state();
        if (!latent) {
            throw std::invalid_argument(
                "CUDA latent attention execution requires latent attention state");
        }
        if (latent->latent_rank > 512) {
            throw std::invalid_argument(
                "CUDA latent attention currently supports latent ranks up to 512");
        }
        if (attention.output_gate.has_value() &&
            compiled.execution.kind != AttentionExecutionKind::FactorizedLatent) {
            throw std::invalid_argument(
                "CUDA latent attention does not support query gates outside factorized execution");
        }
        if (attention.multi_axis_position()) {
            throw std::invalid_argument(
                "CUDA latent attention does not support M-RoPE yet");
        }
    }

    if (attention.rope_position() &&
        std::holds_alternative<LongRopeScaling>(attention.rope_position()->scaling)) {
        const auto& scaling =
            std::get<LongRopeScaling>(attention.rope_position()->scaling);
        if (scaling.short_factors.empty() ||
            scaling.short_factors.size() != scaling.long_factors.size() ||
            scaling.short_factors.size() > 128) {
            throw std::invalid_argument("CUDA LongRoPE requires non-empty factors");
        }
    }
}

inline void validate_cuda_attention_semantics(
    const CompiledModelProgram& program) {
    for (const auto& layer : program.layers) {
        if (const auto* compiled =
                std::get_if<CompiledAttentionProgram>(&layer.mixer)) {
            validate_cuda_attention_semantics(*compiled);
        }
    }
}

}
