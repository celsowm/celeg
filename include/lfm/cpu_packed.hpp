#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <memory>
#include <utility>
#include <vector>

namespace lfm {

class CpuModel;

struct CpuPrefillItem {
    CpuModel* session = nullptr;
    int32_t token = 0;
    bool final_token = false;
};

struct CpuPackedStepMetrics {
    size_t batch_size = 0;
    double elapsed_ms = 0.0;
};

class CpuPackedExecutor {
public:
    CpuPackedExecutor();
    ~CpuPackedExecutor();
    CpuPackedExecutor(CpuPackedExecutor&&) noexcept;
    CpuPackedExecutor& operator=(CpuPackedExecutor&&) noexcept;
    CpuPackedExecutor(const CpuPackedExecutor&) = delete;
    CpuPackedExecutor& operator=(const CpuPackedExecutor&) = delete;

    CpuPackedStepMetrics prefill_step(std::span<const CpuPrefillItem> items);
    CpuPackedStepMetrics prefill_chunk(CpuModel* session,
                                       std::span<const int32_t> tokens,
                                       bool final_chunk);
    std::pair<std::vector<int32_t>, CpuPackedStepMetrics>
    decode_step(std::span<CpuModel* const> sessions);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lfm
