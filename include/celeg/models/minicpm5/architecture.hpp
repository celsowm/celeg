#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_minicpm5_architecture();
void register_minicpm5_architecture(ArchitectureCatalog& catalog);
}
