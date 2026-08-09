#pragma once

#include "celeg/backend/cpu/numa.hpp"

#include <cstddef>

namespace celeg {

// Focused scheduling policy for assigning new sessions to CPU NUMA nodes.
// Allocation and memory binding remain owned by the page/model layers.
class CpuNumaPlacement {
public:
    CpuNumaPlacement(CpuNumaMode mode, CpuNumaTopology topology);

    int next_node();

private:
    CpuNumaMode mode_;
    CpuNumaTopology topology_;
    size_t next_node_ = 0;
};

} // namespace celeg
