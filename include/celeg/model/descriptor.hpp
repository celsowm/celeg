#pragma once

#include "celeg/model/architecture.hpp"

#include <filesystem>

namespace celeg {

// Registers model descriptors loaded from the declarative descriptor store.
// Descriptor evaluation is bounded to metadata, graph construction, and
// validated tensor-binding data; it cannot invoke runtime callbacks.
void register_descriptor_architectures(ArchitectureCatalog& catalog,
                                       const std::filesystem::path& directory);

} // namespace celeg
