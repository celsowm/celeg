#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace celeg {

enum class CpuAffinityPolicy : unsigned char {
    None,
    Compact,
    Scatter,
};

const char* cpu_affinity_name(CpuAffinityPolicy policy);
CpuAffinityPolicy parse_cpu_affinity(const std::string& value);

struct CpuTopology {
    std::vector<int> allowed_cpus;
    size_t numa_nodes = 1;

    std::vector<int> worker_cpu_order(CpuAffinityPolicy policy,
                                      size_t workers) const;
    std::string summary() const;
};

CpuTopology detect_cpu_topology();

}
