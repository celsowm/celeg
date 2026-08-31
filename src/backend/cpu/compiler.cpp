#include "celeg/backend/cpu/compiler.hpp"

#include "celeg/backend/attention_capabilities.hpp"

#include <stdexcept>

namespace celeg {

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
         .relative_position_bias = true});
    validate_moe_backend_capabilities(program, "CPU", {true, true, false, false});
    return program;
}

}
