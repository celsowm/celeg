#include "celeg/backend/cpu/compiler.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CpuModelCompiler::compile(const ResolvedModel& model) const {
    CompiledModelProgram program = build_model_program(model);
    validate_moe_backend_capabilities(program, "CPU", {true, true, false, false});
    return program;
}

}
