#pragma once

#include "celeg/backend/cuda/packed_workspace.hpp"

#include <cstdint>
#include <vector>

namespace celeg {

void stage_packed_persistent_metadata(
    PackedWorkspace& workspace,
    const std::vector<PackedSessionContext>& models,
    const RuntimeTopology& shape);

void stage_packed_step_metadata(
    PackedWorkspace& workspace,
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>* page_tables);

} // namespace celeg
