#pragma once

#include "celeg/model/program.hpp"

namespace celeg {

class CudaModelCompiler {
public:
    CompiledModelProgram compile(const ResolvedModel& model) const;
};

} // namespace celeg
