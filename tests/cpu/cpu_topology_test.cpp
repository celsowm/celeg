#include "celeg/backend/cpu/topology.hpp"
#include "support/assertions.hpp"
#include <iostream>

int main() {
    const celeg::CpuTopology topology = celeg::detect_cpu_topology();
    CELEG_TEST_CHECK(!topology.allowed_cpus.empty());
    const auto compact = topology.worker_cpu_order(celeg::CpuAffinityPolicy::Compact, 4);
    const auto scatter = topology.worker_cpu_order(celeg::CpuAffinityPolicy::Scatter, 4);
    CELEG_TEST_CHECK(compact.size() == 4);
    CELEG_TEST_CHECK(scatter.size() == 4);
    CELEG_TEST_CHECK(celeg::parse_cpu_affinity("none") == celeg::CpuAffinityPolicy::None);
    CELEG_TEST_CHECK(celeg::parse_cpu_affinity("compact") == celeg::CpuAffinityPolicy::Compact);
    CELEG_TEST_CHECK(celeg::parse_cpu_affinity("scatter") == celeg::CpuAffinityPolicy::Scatter);
    std::cout << "cpu_topology_test: " << topology.summary() << '\n';
}
