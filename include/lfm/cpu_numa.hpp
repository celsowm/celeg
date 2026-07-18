#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lfm {

enum class CpuNumaMode : uint8_t {
    Disabled,
    Local,
    ReplicateWeights,
};

const char* cpu_numa_mode_name(CpuNumaMode mode);
CpuNumaMode parse_cpu_numa_mode(const std::string& text);

struct CpuNumaTopology {
    std::vector<std::vector<int>> node_cpus;

    size_t node_count() const { return node_cpus.empty() ? 1 : node_cpus.size(); }
    int node_for_cpu(int cpu) const;
    std::string summary() const;
};

CpuNumaTopology detect_cpu_numa_topology();

// Best-effort Linux placement. Returns false when NUMA placement is unavailable
// or forbidden by the host/container. The memory remains valid in all cases.
bool bind_memory_to_numa_node(void* address, size_t bytes, int node) noexcept;

} // namespace lfm
