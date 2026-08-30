#include "backend/cuda/compiler.hpp"

#include "celeg/backend/moe_capabilities.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    for (const auto& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;
        if (attention.uses_external_memory()) {
            throw std::invalid_argument(
                "CUDA lowering currently supports self-attention sources only");
        }
        if (!attention.has_causal_pattern()) {
            throw std::invalid_argument(
                "CUDA lowering currently supports only causal attention patterns");
        }
        if (!attention.rope_position() &&
            !std::holds_alternative<NoPositionEncodingSpec>(attention.position)) {
            throw std::invalid_argument(
                "CUDA lowering currently supports only RoPE or no position bias");
        }
        if (std::holds_alternative<RelativePositionBiasSpec>(attention.bias)) {
            throw std::invalid_argument(
                "CUDA lowering currently supports ALiBi but not relative-position tables");
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
