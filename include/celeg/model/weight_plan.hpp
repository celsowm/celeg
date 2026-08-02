#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg {

// Builds the canonical tensor requests for a dense transformer. Architecture
// providers retain ownership of naming policies and non-dense variants; this
// helper only consumes the already-resolved topology.
void build_dense_weight_plan(ResolvedModel& model);

} // namespace celeg
