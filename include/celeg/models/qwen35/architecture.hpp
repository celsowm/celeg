#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {
std::unique_ptr<IArchitecture> make_qwen35_architecture();
void register_qwen35_architecture(ArchitectureCatalog& catalog);
}
