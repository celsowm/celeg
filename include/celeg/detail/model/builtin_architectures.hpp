#pragma once

#include "celeg/model/architecture.hpp"

#include <memory>

namespace celeg::detail {

std::unique_ptr<IArchitecture> make_lfm2_architecture();
std::unique_ptr<IArchitecture> make_granite_architecture();

} // namespace celeg::detail
