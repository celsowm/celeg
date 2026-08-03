#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {

std::unique_ptr<IArchitecture> make_lfm2_architecture();
std::unique_ptr<IArchitecture> make_granite_architecture();
std::unique_ptr<IArchitecture> make_gemma4_architecture();
std::unique_ptr<IArchitecture> make_minicpm5_architecture();
std::unique_ptr<IArchitecture> make_smollm3_architecture();

} // namespace celeg::detail
