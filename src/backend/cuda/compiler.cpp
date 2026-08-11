#include "celeg/backend/cuda/compiler.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    if (!model.capabilities.supports_cuda) {
        throw std::invalid_argument("resolved model does not support CUDA");
    }
    CompiledModelProgram program = build_model_program(model);
    for (const auto& layer : program.layers) {
        if (layer.attention && layer.attention->uses_external_memory()) {
            throw std::invalid_argument(
                "CUDA lowering currently supports self-attention sources only");
        }
        if (layer.attention && !layer.attention->has_causal_pattern()) {
            throw std::invalid_argument(
                "CUDA lowering currently supports only causal attention patterns");
        }
        if (layer.attention &&
            !layer.attention->rope_position() &&
            !std::holds_alternative<NoPositionEncodingSpec>(layer.attention->position)) {
            throw std::invalid_argument(
                "CUDA lowering currently supports only RoPE or no position bias");
        }
        if (layer.attention &&
            std::holds_alternative<RelativePositionBiasSpec>(layer.attention->bias)) {
            throw std::invalid_argument(
                "CUDA lowering currently supports ALiBi but not relative-position tables");
        }
        if (layer.attention && layer.attention->uses_latent_state()) {
            const auto& latent = *layer.attention->latent_state();
            if (latent.latent_rank > 512) {
                throw std::invalid_argument(
                    "CUDA latent attention currently supports latent ranks up to 512");
            }
            if ((layer.attention->output_gate.enabled() && !latent.factorized) ||
                layer.attention->multi_axis_position()) {
                throw std::invalid_argument(
                    "CUDA latent attention does not support query gates or M-RoPE yet");
            }
        }
        if (layer.attention && layer.attention->rope_position() &&
            layer.attention->rope_position()->scaling.kind == RopeScalingKind::Long) {
            const auto& scaling = layer.attention->rope_position()->scaling;
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

} // namespace celeg
