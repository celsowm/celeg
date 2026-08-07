#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_smollm3_architecture();
void register_smollm3_architecture(ArchitectureCatalog& catalog);
}
