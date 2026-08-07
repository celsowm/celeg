#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_nemotron_h_architecture();
void register_nemotron_h_architecture(ArchitectureCatalog& catalog);
}
