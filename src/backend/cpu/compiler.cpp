#include "celeg/backend/cpu/compiler.hpp"

#include <stdexcept>

namespace celeg {

CompiledModelProgram CpuModelCompiler::compile(const ResolvedModel& model) const {
    if (!model.capabilities.supports_cpu) {
        throw std::invalid_argument("resolved model does not support CPU");
    }
    return build_model_program(model);
}

} // namespace celeg
