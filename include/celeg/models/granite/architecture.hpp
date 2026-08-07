#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_granite_architecture();
void register_granite_architecture(ArchitectureCatalog& catalog);
}
