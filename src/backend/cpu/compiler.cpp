#include "celeg/backend/cpu/compiler.hpp"

#include "celeg/backend/attention_capabilities.hpp"

#include <stdexcept>

namespace celeg {
namespace {

bool cpu_state_scalar_supported(StateScalarType scalar) {
    return scalar == StateScalarType::FP32 || scalar == StateScalarType::BF16;
}

void validate_cpu_attention_semantics(const CompiledModelProgram& program) {
    for (const CompiledLayerProgram& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& attention = compiled->semantics;
        if (const auto* ordinary = std::get_if<OrdinaryKvStateSpec>(&attention.state)) {
            if (!cpu_state_scalar_supported(ordinary->storage.key) ||
                !cpu_state_scalar_supported(ordinary->storage.value)) {
                throw std::invalid_argument(
                    "CPU attention state currently supports FP32 or BF16 ordinary KV only");
            }
        } else if (const auto* latent =
                       std::get_if<LatentAttentionStateSpec>(&attention.state)) {
            if (!cpu_state_scalar_supported(latent->storage.latent) ||
                !cpu_state_scalar_supported(latent->storage.rotary)) {
                throw std::invalid_argument(
                    "CPU latent attention state currently supports FP32 or BF16 storage only");
            }
            if (attention.multi_axis_position()) {
                throw std::invalid_argument(
                    "CPU latent attention does not yet support multi-axis RoPE");
            }
        }
    }
}

}

CompiledModelProgram CpuModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    validate_attention_backend_capabilities(
        program, "CPU",
        {.full_causal = true,
         .sliding_window = true,
         .bidirectional = true,
         .prefix_lm = true,
         .block_sparse = true,
         .dynamic_sparse = true,
         .external_memory = false,
         .alibi = true,
         .relative_position_bias = true,
         .no_position = true,
         .rope = true,
         .multi_axis_rope = true,
         .standard_execution = true,
         .latent_execution = true,
         .factorized_latent_execution = true});
    validate_cpu_attention_semantics(program);
    validate_moe_backend_capabilities(program, "CPU", {true, true, false, false});
    return program;
}

}
