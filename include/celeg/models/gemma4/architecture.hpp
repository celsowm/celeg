#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_gemma4_architecture();
void register_gemma4_architecture(ArchitectureCatalog& catalog);
}
