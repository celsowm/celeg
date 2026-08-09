#pragma once

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/backend/cpu/model.hpp"

namespace celeg {

// Converts execution outcomes into scheduler throughput/timing metrics. The
// collector does not inspect request state or make scheduling decisions.
class CpuMetricsCollector {
public:
    explicit CpuMetricsCollector(CpuConcurrentMetrics& metrics)
        : metrics_(metrics) {}

    void record_decode_batch(size_t tokens, const CpuBatchMetrics& outcome);
    void record_ragged_prefill(size_t tokens, const CpuBatchMetrics& outcome);
    void record_chunked_prefill(size_t tokens, const CpuBatchMetrics& outcome);
    void record_scheduler(double elapsed_ms);
    void record_prefill_tokens(size_t tokens);

private:
    CpuConcurrentMetrics& metrics_;
};

} // namespace celeg
