#include "celeg/backend/cpu/numa.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace celeg {

const char* cpu_numa_mode_name(CpuNumaMode mode) {
    switch (mode) {
        case CpuNumaMode::Disabled: return "disabled";
        case CpuNumaMode::Local: return "local";
        case CpuNumaMode::ReplicateWeights: return "replicate-weights";
    }
    return "unknown";
}

CpuNumaMode parse_cpu_numa_mode(const std::string& text) {
    if (text == "disabled" || text == "off" || text == "none") {
        return CpuNumaMode::Disabled;
    }
    if (text == "local") return CpuNumaMode::Local;
    if (text == "replicate" || text == "replicate-weights") {
        return CpuNumaMode::ReplicateWeights;
    }
    throw std::invalid_argument("unknown CPU NUMA mode: " + text);
}

namespace {
std::vector<int> parse_cpu_list(const std::string& text) {
    std::vector<int> cpus;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        const auto dash = token.find('-');
        if (dash == std::string::npos) {
            if (!token.empty()) cpus.push_back(std::stoi(token));
            continue;
        }
        const int first = std::stoi(token.substr(0, dash));
        const int last = std::stoi(token.substr(dash + 1));
        for (int cpu = first; cpu <= last; ++cpu) cpus.push_back(cpu);
    }
    return cpus;
}
}

CpuNumaTopology detect_cpu_numa_topology() {
    CpuNumaTopology result;
#if defined(__linux__)
    const std::filesystem::path base("/sys/devices/system/node");
    std::error_code ec;
    if (std::filesystem::exists(base, ec)) {
        std::vector<std::pair<int, std::vector<int>>> indexed;
        for (const auto& entry : std::filesystem::directory_iterator(base, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("node", 0) != 0 || name.size() <= 4) continue;
            if (!std::all_of(name.begin() + 4, name.end(), [](char c) {
                    return c >= '0' && c <= '9';
                })) continue;
            const int node = std::stoi(name.substr(4));
            std::ifstream input(entry.path() / "cpulist");
            std::string cpulist;
            std::getline(input, cpulist);
            indexed.emplace_back(node, parse_cpu_list(cpulist));
        }
        std::sort(indexed.begin(), indexed.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (auto& [_, cpus] : indexed) result.node_cpus.push_back(std::move(cpus));
    }
#endif
    if (result.node_cpus.empty()) result.node_cpus.push_back({});
    return result;
}

int CpuNumaTopology::node_for_cpu(int cpu) const {
    for (size_t node = 0; node < node_cpus.size(); ++node) {
        if (std::find(node_cpus[node].begin(), node_cpus[node].end(), cpu) !=
            node_cpus[node].end()) {
            return static_cast<int>(node);
        }
    }
    return 0;
}

std::string CpuNumaTopology::summary() const {
    std::ostringstream out;
    out << "numa-nodes=" << node_count();
    for (size_t node = 0; node < node_cpus.size(); ++node) {
        out << " node" << node << "-cpus=" << node_cpus[node].size();
    }
    return out.str();
}

bool bind_memory_to_numa_node(void* address, size_t bytes, int node) noexcept {
#if defined(__linux__) && defined(SYS_mbind)
    if (!address || bytes == 0 || node < 0) return false;
    constexpr size_t bits = sizeof(unsigned long) * 8;
    const size_t words = static_cast<size_t>(node) / bits + 1;
    std::vector<unsigned long> mask(words, 0);
    mask[static_cast<size_t>(node) / bits] |=
        1UL << (static_cast<size_t>(node) % bits);
    const unsigned long maxnode = static_cast<unsigned long>(words * bits);
    const long result = syscall(SYS_mbind, address, bytes, MPOL_BIND,
                                mask.data(), maxnode, MPOL_MF_MOVE);
    return result == 0;
#else
    (void)address; (void)bytes; (void)node;
    return false;
#endif
}

} // namespace celeg
