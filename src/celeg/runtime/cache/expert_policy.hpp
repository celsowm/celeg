#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace celeg {

struct ExpertKey {
    int layer = -1;
    int expert = -1;

    void validate() const {
        if (layer < 0 || expert < 0) {
            throw std::invalid_argument("expert key must be non-negative");
        }
    }

    std::uint64_t packed() const {
        validate();
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(layer)) << 32) |
               static_cast<std::uint32_t>(expert);
    }

    friend bool operator==(const ExpertKey&, const ExpertKey&) = default;
};

struct ExpertCacheMetrics {
    std::size_t hits = 0;
    std::size_t true_misses = 0;
    std::size_t coalesced_waits = 0;
    std::size_t evictions = 0;
    std::size_t bytes_read = 0;
    double wait_time_ms = 0.0;
};

struct ExpertBudget {
    std::size_t bytes = 0;
    std::size_t payload_bytes = 0;

    std::size_t capacity() const {
        if (bytes == 0 || payload_bytes == 0) return 0;
        return bytes / payload_bytes;
    }

    void validate() const {
        if (payload_bytes == 0) {
            throw std::invalid_argument("expert payload budget has zero payload size");
        }
    }
};

inline double expert_priority(double heat, std::uint64_t age) {
    return heat + 1.0 / (1.0 + static_cast<double>(age));
}

}
