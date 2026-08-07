#include "celeg/backend/cpu/compiler.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CpuModelCompiler::compile(const ResolvedModel& model) const {
    if (!model.capabilities.supports_cpu) {
        throw std::invalid_argument("resolved model does not support CPU");
    }
    CompiledModelProgram program = build_model_program(model);
    validate_moe_backend_capabilities(program, "CPU", {false, true, false, false});
    return program;
}

} // namespace celeg
