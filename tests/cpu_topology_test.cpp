#include "lfm/backend/cpu/topology.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    const lfm::CpuTopology topology = lfm::detect_cpu_topology();
    LFM_TEST_CHECK(!topology.allowed_cpus.empty());
    const auto compact = topology.worker_cpu_order(lfm::CpuAffinityPolicy::Compact, 4);
    const auto scatter = topology.worker_cpu_order(lfm::CpuAffinityPolicy::Scatter, 4);
    LFM_TEST_CHECK(compact.size() == 4);
    LFM_TEST_CHECK(scatter.size() == 4);
    LFM_TEST_CHECK(lfm::parse_cpu_affinity("none") == lfm::CpuAffinityPolicy::None);
    LFM_TEST_CHECK(lfm::parse_cpu_affinity("compact") == lfm::CpuAffinityPolicy::Compact);
    LFM_TEST_CHECK(lfm::parse_cpu_affinity("scatter") == lfm::CpuAffinityPolicy::Scatter);
    std::cout << "cpu_topology_test: " << topology.summary() << '\n';
}
