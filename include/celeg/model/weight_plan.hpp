#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg {

// `repository` is the checkpoint's weight repository, used to pick the
// naming-policy candidate that actually exists on disk (rather than
// blindly trusting the first candidate template) and, for MoE layers, to
// decide the routed-expert checkpoint's physical layout (packed vs
// individual) once, here, instead of leaving backends to re-derive it from
// tensor-name spellings during setup. Pass nullptr only when no checkpoint
// is available (e.g. synthetic graphs in tests); in that case candidate
// selection degrades to "first candidate" and MoE layers are always
// planned as individual-expert.
void build_weight_plan_from_graph(ResolvedModel& model,
                                  const ITensorNamingPolicy& naming_policy,
                                  const IWeightRepository* repository);
void resolve_weight_plan(ResolvedModel& model,
                         const ITensorNamingPolicy& naming_policy,
                         const IWeightRepository* repository);

}
