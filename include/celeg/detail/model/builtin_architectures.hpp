#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {

std::unique_ptr<IArchitecture> make_lfm2_architecture();
std::unique_ptr<IArchitecture> make_granite_architecture();
std::unique_ptr<IArchitecture> make_gemma4_architecture();
std::unique_ptr<IArchitecture> make_minicpm5_architecture();
std::unique_ptr<IArchitecture> make_smollm3_architecture();

void register_lfm2_architecture(ArchitectureCatalog& catalog);
void register_granite_architecture(ArchitectureCatalog& catalog);
void register_gemma4_architecture(ArchitectureCatalog& catalog);
void register_minicpm5_architecture(ArchitectureCatalog& catalog);
void register_smollm3_architecture(ArchitectureCatalog& catalog);

// Single composition boundary for the built-in family bundles. Individual
// families own their registration details; the runtime only installs the
// bundle as one catalog extension.
void register_builtin_family_bundles(ArchitectureCatalog& catalog);

} // namespace celeg::detail
