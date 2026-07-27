#include "lfm/backend/cpu/numa.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    assert(lfm::parse_cpu_numa_mode("disabled") == lfm::CpuNumaMode::Disabled);
    assert(lfm::parse_cpu_numa_mode("local") == lfm::CpuNumaMode::Local);
    assert(lfm::parse_cpu_numa_mode("replicate") == lfm::CpuNumaMode::ReplicateWeights);
    const auto topology = lfm::detect_cpu_numa_topology();
    assert(topology.node_count() >= 1);
    std::vector<unsigned char> memory(4096);
    (void)lfm::bind_memory_to_numa_node(memory.data(), memory.size(), 0);
    std::cout << "cpu_numa_test: ok " << topology.summary() << '\n';
}
