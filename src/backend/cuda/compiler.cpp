#include "backend/cuda/compiler.hpp"

#include "backend/cuda/attention_semantic_capability.hpp"
#include "celeg/backend/attention_capabilities.hpp"
#include "celeg/backend/moe_capabilities.hpp"

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    validate_attention_backend_capabilities(
        program, "CUDA",
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
    validate_cuda_attention_semantics(program);
    validate_moe_backend_capabilities(program, "CUDA", {true, true, false, false});
    return program;
}

}
