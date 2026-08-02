#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/resolved.hpp"

namespace celeg::detail {

RuntimeTopology resolve_granite_topology(const CheckpointMetadata& metadata);

} // namespace celeg::detail
