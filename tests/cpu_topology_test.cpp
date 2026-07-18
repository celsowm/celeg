#include "lfm/cpu_topology.hpp"

#include <cassert>
#include <iostream>

int main() {
    const lfm::CpuTopology topology = lfm::detect_cpu_topology();
    assert(!topology.allowed_cpus.empty());
    const auto compact = topology.worker_cpu_order(lfm::CpuAffinityPolicy::Compact, 4);
    const auto scatter = topology.worker_cpu_order(lfm::CpuAffinityPolicy::Scatter, 4);
    assert(compact.size() == 4);
    assert(scatter.size() == 4);
    assert(lfm::parse_cpu_affinity("none") == lfm::CpuAffinityPolicy::None);
    assert(lfm::parse_cpu_affinity("compact") == lfm::CpuAffinityPolicy::Compact);
    assert(lfm::parse_cpu_affinity("scatter") == lfm::CpuAffinityPolicy::Scatter);
    std::cout << "cpu_topology_test: " << topology.summary() << '\n';
}
