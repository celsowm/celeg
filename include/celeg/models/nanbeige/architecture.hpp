#pragma once

#include "celeg/model/architecture.hpp"

namespace celeg::detail {

std::unique_ptr<IArchitecture> make_nanbeige42_architecture();
void register_nanbeige42_architecture(ArchitectureCatalog& catalog);

} // namespace celeg::detail
