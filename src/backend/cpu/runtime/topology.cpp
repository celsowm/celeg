#include "celeg/backend/cpu/topology.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace celeg {

const char* cpu_affinity_name(CpuAffinityPolicy policy) {
    switch (policy) {
        case CpuAffinityPolicy::None: return "none";
        case CpuAffinityPolicy::Compact: return "compact";
        case CpuAffinityPolicy::Scatter: return "scatter";
    }
    return "unknown";
}

CpuAffinityPolicy parse_cpu_affinity(const std::string& value) {
    if (value == "none") return CpuAffinityPolicy::None;
    if (value == "compact") return CpuAffinityPolicy::Compact;
    if (value == "scatter") return CpuAffinityPolicy::Scatter;
    throw std::invalid_argument("unknown CPU affinity policy: " + value);
}

CpuTopology detect_cpu_topology() {
    CpuTopology result;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) result.allowed_cpus.push_back(cpu);
        }
    }
#endif
    if (result.allowed_cpus.empty()) {
        const unsigned count = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned cpu = 0; cpu < count; ++cpu) {
            result.allowed_cpus.push_back(static_cast<int>(cpu));
        }
    }
#if defined(__linux__)
    size_t nodes = 0;
    std::error_code ec;
    const std::filesystem::path base("/sys/devices/system/node");
    if (std::filesystem::exists(base, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.size() > 4 && name.rfind("node", 0) == 0 &&
                std::all_of(name.begin() + 4, name.end(), [](char c) {
                    return c >= '0' && c <= '9';
                })) {
                ++nodes;
            }
        }
    }
    result.numa_nodes = std::max<size_t>(1, nodes);
#endif
    return result;
}

std::vector<int> CpuTopology::worker_cpu_order(CpuAffinityPolicy policy,
                                                size_t workers) const {
    std::vector<int> result;
    if (policy == CpuAffinityPolicy::None || workers == 0 ||
        allowed_cpus.empty()) {
        return result;
    }
    result.reserve(workers);
    if (policy == CpuAffinityPolicy::Compact) {
        for (size_t i = 0; i < workers; ++i) {
            result.push_back(allowed_cpus[i % allowed_cpus.size()]);
        }
        return result;
    }
    // Spread workers across the process affinity mask. This is deliberately
    // topology-light and works in containers where physical package metadata
    // is often hidden.
    for (size_t i = 0; i < workers; ++i) {
        const size_t index = std::min(
            allowed_cpus.size() - 1,
            (i * allowed_cpus.size()) / workers);
        result.push_back(allowed_cpus[index]);
    }
    return result;
}

std::string CpuTopology::summary() const {
    std::ostringstream out;
    out << "allowed-cpus=" << allowed_cpus.size()
        << " numa-nodes=" << numa_nodes;
    if (!allowed_cpus.empty()) {
        out << " first-cpu=" << allowed_cpus.front()
            << " last-cpu=" << allowed_cpus.back();
    }
    return out.str();
}

} // namespace celeg
