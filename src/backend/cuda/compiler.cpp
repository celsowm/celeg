#include "backend/cuda/compiler.hpp"

#include "celeg/backend/attention_capabilities.hpp"
#include "celeg/backend/moe_capabilities.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    validate_attention_backend_capabilities(program, "CUDA", {
        true,  // full_causal
        true,  // sliding_window
        true,  // bidirectional
        true,  // prefix_lm
        true,  // block_sparse
        true,  // dynamic_sparse
        false, // external_memory
        true,  // alibi
        false, // relative_position_bias
    });

    for (const auto& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;
        const bool bidirectional =
            std::holds_alternative<BidirectionalPattern>(attention.pattern);
        const auto* prefix_lm = std::get_if<PrefixLmPattern>(&attention.pattern);
        const auto* block_sparse = std::get_if<BlockSparsePattern>(&attention.pattern);
        const auto* dynamic_sparse = std::get_if<DynamicSparsePattern>(&attention.pattern);
        if (prefix_lm && prefix_lm->prefix_length <= 0) {
            throw std::invalid_argument(
                "CUDA prefix-LM attention requires a positive prefix length");
        }
        if (block_sparse &&
            (block_sparse->block_size <= 0 || block_sparse->local_blocks <= 0 ||
             block_sparse->global_blocks < 0)) {
            throw std::invalid_argument(
                "CUDA block-sparse attention requires positive block/local sizes and non-negative global blocks");
        }
        if (dynamic_sparse &&
            (dynamic_sparse->block_size <= 0 ||
             dynamic_sparse->max_selected_blocks <= 0 ||
             dynamic_sparse->max_selected_blocks > 32)) {
            throw std::invalid_argument(
                "CUDA dynamic-sparse attention requires a positive block size and 1..32 selected blocks");
        }
        const bool constrained_standard_pattern =
            bidirectional || prefix_lm != nullptr || block_sparse != nullptr ||
            dynamic_sparse != nullptr;
        if (constrained_standard_pattern &&
            !std::holds_alternative<NoAttentionBiasSpec>(attention.bias)) {
            throw std::invalid_argument(
                "CUDA constrained attention patterns currently support no attention bias");
        }
        if (constrained_standard_pattern &&
            compiled->execution.kind != AttentionExecutionKind::Standard) {
            throw std::invalid_argument(
                "CUDA constrained attention patterns currently support standard attention only");
        }
        if (!attention.rope_position() &&
            !std::holds_alternative<NoPositionEncodingSpec>(attention.position)) {
            throw std::invalid_argument(
                "CUDA lowering currently supports only RoPE or no position bias");
        }
        if (compiled->execution.kind != AttentionExecutionKind::Standard) {
            const auto& latent = *attention.latent_state();
            if (latent.latent_rank > 512) {
                throw std::invalid_argument(
                    "CUDA latent attention currently supports latent ranks up to 512");
            }
            if ((attention.output_gate.has_value() &&
                 compiled->execution.kind != AttentionExecutionKind::FactorizedLatent) ||
                attention.multi_axis_position()) {
                throw std::invalid_argument(
                    "CUDA latent attention does not support query gates or M-RoPE yet");
            }
        }
        if (attention.rope_position() &&
            std::holds_alternative<LongRopeScaling>(attention.rope_position()->scaling)) {
            const auto& scaling = std::get<LongRopeScaling>(attention.rope_position()->scaling);
            if (scaling.short_factors.empty() ||
                scaling.short_factors.size() != scaling.long_factors.size() ||
                scaling.short_factors.size() > 128) {
                throw std::invalid_argument("CUDA LongRoPE requires non-empty factors");
            }
        }
    }
    validate_moe_backend_capabilities(program, "CUDA", {true, true, false, false});
    return program;
}

}
