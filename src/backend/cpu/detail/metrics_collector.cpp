#include "metrics_collector.hpp"

#include <algorithm>

namespace celeg {

void CpuMetricsCollector::record_decode_batch(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++metrics_.packed_decode_steps;
    metrics_.decoded_tokens += tokens;
    metrics_.packed_decode_tokens += tokens;
    metrics_.maximum_packed_batch = std::max<uint64_t>(
        metrics_.maximum_packed_batch, tokens);
    metrics_.cumulative_packed_decode_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_ragged_prefill(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++metrics_.ragged_prefill_steps;
    metrics_.prefill_tokens += tokens;
    metrics_.ragged_prefill_tokens += tokens;
    metrics_.maximum_ragged_prefill_batch = std::max<uint64_t>(
        metrics_.maximum_ragged_prefill_batch, tokens);
    metrics_.cumulative_ragged_prefill_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_chunked_prefill(
    size_t tokens, const CpuBatchMetrics& outcome) {
    ++extras_.chunked_prefill_steps;
    metrics_.direct_paged_prefill_tokens += tokens;
    metrics_.prefill_tokens += tokens;
    extras_.maximum_prefill_chunk = std::max<uint64_t>(
        extras_.maximum_prefill_chunk, tokens);
    extras_.cumulative_direct_prefill_ms += outcome.elapsed_ms;
}

void CpuMetricsCollector::record_scheduler(double elapsed_ms) {
    ++metrics_.scheduler_steps;
    metrics_.cumulative_step_ms += elapsed_ms;
}

void CpuMetricsCollector::record_prefill_tokens(size_t tokens) {
    metrics_.prefill_tokens += tokens;
    metrics_.lane_prefill_tokens += tokens;
}

} // namespace celeg
