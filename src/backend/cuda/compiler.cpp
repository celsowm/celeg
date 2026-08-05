#include "celeg/backend/cuda/compiler.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CudaModelCompiler::compile(const ResolvedModel& model) const {
    if (!model.capabilities.supports_cuda) {
        throw std::invalid_argument("resolved model does not support CUDA");
    }
    CompiledModelProgram program = build_model_program(model);
    validate_moe_backend_capabilities(program, "CUDA", {});
    return program;
}

} // namespace celeg
