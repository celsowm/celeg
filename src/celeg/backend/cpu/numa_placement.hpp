#pragma once

#include "celeg/backend/cpu/numa.hpp"

#include <cstddef>

namespace celeg {

class CpuNumaPlacement {
public:
    CpuNumaPlacement(CpuNumaMode mode, CpuNumaTopology topology);

    int next_node();

private:
    CpuNumaMode mode_;
    CpuNumaTopology topology_;
    size_t next_node_ = 0;
};

}
