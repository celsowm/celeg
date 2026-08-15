#pragma once

#include "expert_cache.hpp"

#include "celeg/backend/cpu/quantization.hpp"
#include "celeg/runtime/cache/expert_policy.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

struct CpuExpertBackingMetrics {
    std::size_t resident_bytes = 0;
    std::size_t hits = 0;
    std::size_t true_misses = 0;
    std::size_t coalesced_waits = 0;
    std::size_t evictions = 0;
};

class CpuExpertBackingStore {
public:
    CpuExpertBackingStore(const std::filesystem::path& pack_path,
                          std::size_t budget_bytes,
                          int hidden_size,
                          int intermediate_size);

    std::shared_ptr<const CpuExpertWeights> acquire(ExpertKey key);
    CpuExpertBackingMetrics metrics() const;

private:
    std::unique_ptr<CpuPackReader> reader_;
    std::unique_ptr<CpuExpertCache> cache_;
    int hidden_size_ = 0;
    int intermediate_size_ = 0;
};

}
