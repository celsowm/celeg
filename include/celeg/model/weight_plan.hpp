#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg {

void build_weight_plan_from_graph(ResolvedModel& model,
                                  const ITensorNamingPolicy& naming_policy);
void resolve_weight_plan(ResolvedModel& model,
                         const ITensorNamingPolicy& naming_policy);

}
