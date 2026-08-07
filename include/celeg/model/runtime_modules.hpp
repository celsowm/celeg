#pragma once

#include "celeg/runtime/providers.hpp"

#include <memory>
#include <vector>

namespace celeg {

class ArchitectureCatalog;

std::vector<std::unique_ptr<IRuntimeModule>> make_builtin_runtime_modules();
void add_builtin_architectures(ArchitectureCatalog& catalog);

} // namespace celeg
