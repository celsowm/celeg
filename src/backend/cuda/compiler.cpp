#include "backend/cuda/compiler.hpp"

#include "backend/cuda/attention_semantic_capability.hpp"
#include "celeg/backend/cuda/attention_capabilities.hpp"
#include "celeg/backend/moe_capabilities.hpp"

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    validate_attention_backend_capabilities(
        program, "CUDA", cuda_attention_capabilities());
    validate_cuda_attention_semantics(program);
    validate_moe_backend_capabilities(program, "CUDA", {true, true, false, false});
    return program;
}

}
