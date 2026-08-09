#include "celeg/backend/cpu/numa.hpp"
#include "celeg/backend/cpu/numa_placement.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <vector>

int main() {
    CELEG_TEST_CHECK(celeg::parse_cpu_numa_mode("disabled") == celeg::CpuNumaMode::Disabled);
    CELEG_TEST_CHECK(celeg::parse_cpu_numa_mode("local") == celeg::CpuNumaMode::Local);
    CELEG_TEST_CHECK(celeg::parse_cpu_numa_mode("replicate") == celeg::CpuNumaMode::ReplicateWeights);
    const auto topology = celeg::detect_cpu_numa_topology();
    CELEG_TEST_CHECK(topology.node_count() >= 1);
    celeg::CpuNumaPlacement disabled(celeg::CpuNumaMode::Disabled, topology);
    CELEG_TEST_CHECK(disabled.next_node() == -1);
    std::vector<unsigned char> memory(4096);
    (void)celeg::bind_memory_to_numa_node(memory.data(), memory.size(), 0);
    std::cout << "cpu_numa_test: ok " << topology.summary() << '\n';
}
