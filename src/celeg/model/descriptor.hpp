#pragma once

#include "celeg/model/architecture.hpp"

#include <filesystem>

namespace celeg {

void register_descriptor_architectures(ArchitectureCatalog& catalog,
                                       const std::filesystem::path& directory);

}
