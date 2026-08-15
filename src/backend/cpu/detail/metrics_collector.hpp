#pragma once

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/runtime/concurrency/metrics.hpp"

namespace celeg {

class CpuMetricsCollector {
public:
    CpuMetricsCollector(ConcurrentMetrics& metrics, CpuConcurrentMetricsExtras& extras)
        : metrics_(metrics), extras_(extras) {}

    void record_decode_batch(size_t tokens, const CpuBatchMetrics& outcome);
    void record_ragged_prefill(size_t tokens, const CpuBatchMetrics& outcome);
    void record_chunked_prefill(size_t tokens, const CpuBatchMetrics& outcome);
    void record_scheduler(double elapsed_ms);
    void record_prefill_tokens(size_t tokens);

private:
    ConcurrentMetrics& metrics_;
    CpuConcurrentMetricsExtras& extras_;
};

}
