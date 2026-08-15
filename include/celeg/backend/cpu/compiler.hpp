#pragma once

#include "celeg/backend/moe_capabilities.hpp"
#include "celeg/model/program.hpp"

namespace celeg {

class CpuModelCompiler {
public:
    CompiledModelProgram compile(const ResolvedModel& model) const;
};

} // namespace celeg
