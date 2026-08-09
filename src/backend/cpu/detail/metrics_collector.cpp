#include "metrics_collector.hpp"

#include <algorithm>

namespace celeg {

void CpuMetricsCollector::record_decode_batch(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++metrics_.packed_decode_steps;
    metrics_.decode_tokens += tokens;
    metrics_.maximum_decode_batch = std::max<uint64_t>(
        metrics_.maximum_decode_batch, tokens);
    metrics_.cumulative_decode_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_ragged_prefill(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++metrics_.ragged_prefill_steps;
    metrics_.prefill_tokens += tokens;
    metrics_.maximum_prefill_batch = std::max<uint64_t>(
        metrics_.maximum_prefill_batch, tokens);
    metrics_.cumulative_prefill_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_chunked_prefill(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++metrics_.chunked_prefill_steps;
    metrics_.chunked_prefill_tokens += tokens;
    metrics_.prefill_tokens += tokens;
    metrics_.maximum_prefill_chunk = std::max<uint64_t>(
        metrics_.maximum_prefill_chunk, tokens);
    metrics_.cumulative_prefill_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_scheduler(double elapsed_ms) {
    metrics_.cumulative_scheduler_ms += elapsed_ms;
}

void CpuMetricsCollector::record_prefill_tokens(size_t tokens) {
    metrics_.prefill_tokens += tokens;
}

} // namespace celeg
