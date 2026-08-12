#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg {

// Builds canonical tensor requests for every supported semantic layer variant.
// Per-layer shapes come from the final graph; topology contributes only
// checkpoint-layer mapping and import vocabulary facts.
void build_weight_plan_from_graph(ResolvedModel& model,
                                  const ITensorNamingPolicy& naming_policy);
void resolve_weight_plan(ResolvedModel& model,
                         const ITensorNamingPolicy& naming_policy);

} // namespace celeg
