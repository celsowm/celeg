#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {

std::unique_ptr<IArchitecture> make_lfm2_architecture();
std::unique_ptr<IArchitecture> make_granite_architecture();
std::unique_ptr<IArchitecture> make_gemma4_architecture();
std::unique_ptr<IArchitecture> make_minicpm5_architecture();
std::unique_ptr<IArchitecture> make_smollm3_architecture();
std::unique_ptr<IArchitecture> make_qwen35_architecture();
std::unique_ptr<IArchitecture> make_nemotron_h_architecture();

void register_lfm2_architecture(ArchitectureCatalog& catalog);
void register_granite_architecture(ArchitectureCatalog& catalog);
void register_gemma4_architecture(ArchitectureCatalog& catalog);
void register_minicpm5_architecture(ArchitectureCatalog& catalog);
void register_smollm3_architecture(ArchitectureCatalog& catalog);
void register_qwen35_architecture(ArchitectureCatalog& catalog);
void register_nemotron_h_architecture(ArchitectureCatalog& catalog);

} // namespace celeg::detail
