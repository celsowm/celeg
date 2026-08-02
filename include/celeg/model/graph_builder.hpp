#pragma once

#include "celeg/model/resolved.hpp"

namespace celeg {

// Builds the execution graph shared by dense transformer architectures. The
// provider remains responsible for resolving topology and metadata; this
// helper only materializes the already-resolved dense layer contract.
void build_dense_transformer_graph(ResolvedModel& model);

} // namespace celeg
