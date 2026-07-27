#include "lfm/runtime/concurrency/metrics.hpp"

namespace lfm {

GroupedConcurrentMetrics group_concurrent_metrics(
    const ConcurrentMetrics& flat) {
    GroupedConcurrentMetrics grouped;
    grouped.requests = {
        flat.submitted,
        flat.completed,
        flat.cancelled,
        flat.failed,
        flat.active_requests,
        flat.queued_requests,
        flat.average_ttft_ms(),
        flat.average_itl_ms(),
    };
    grouped.scheduler = {
        flat.scheduler_steps,
        flat.cumulative_step_ms,
    };
    grouped.prefill = {
        flat.prefill_tokens,
        flat.direct_paged_prefill_tokens,
        flat.ragged_prefill_steps,
        flat.ragged_prefill_tokens,
        flat.lane_prefill_tokens,
        flat.maximum_ragged_prefill_batch,
        flat.cumulative_ragged_prefill_ms,
    };
    grouped.decode = {
        flat.decoded_tokens,
        flat.packed_decode_steps,
        flat.packed_decode_tokens,
        flat.lane_decode_tokens,
        flat.packed_fallback_batches,
        flat.maximum_packed_batch,
        flat.segmented_paged_decode_steps,
        flat.segmented_paged_decode_tokens,
        flat.cumulative_packed_decode_ms,
    };
    grouped.prefix = {
        flat.prefix_cache_hits,
        flat.prefix_cache_misses,
        flat.prefix_cache_inserts,
        flat.prefix_cache_evictions,
        flat.prefix_cache_partial_hits,
        flat.prefix_reused_tokens,
        flat.prefix_cow_pages,
        flat.prefix_radix_nodes,
        flat.prefix_radix_lookups,
        flat.prefix_cow_bytes_copied,
        flat.prefix_cow_bytes_saved,
    };
    grouped.kv = {
        flat.logical_pages_used,
        flat.logical_pages_total,
        flat.physical_kv_bytes,
    };
    return grouped;
}

} // namespace lfm
