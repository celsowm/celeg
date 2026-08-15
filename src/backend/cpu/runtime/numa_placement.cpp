#include "celeg/backend/cpu/numa_placement.hpp"

#include <utility>

namespace celeg {

CpuNumaPlacement::CpuNumaPlacement(CpuNumaMode mode, CpuNumaTopology topology)
    : mode_(mode), topology_(std::move(topology)) {}

int CpuNumaPlacement::next_node() {
    if (mode_ == CpuNumaMode::Disabled || topology_.node_count() <= 1) return -1;
    const int node = static_cast<int>(next_node_ % topology_.node_count());
    ++next_node_;
    return node;
}

}
