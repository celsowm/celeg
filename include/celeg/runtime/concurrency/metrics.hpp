#pragma once

#include <cstdint>

namespace celeg {

struct ConcurrentMetrics {
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t cancelled = 0;
    uint64_t failed = 0;
    uint64_t scheduler_steps = 0;
    uint64_t prefill_tokens = 0;
    uint64_t decoded_tokens = 0;
    uint64_t active_requests = 0;
    uint64_t queued_requests = 0;
    uint64_t logical_pages_used = 0;
    uint64_t logical_pages_total = 0;
    uint64_t ttft_samples = 0;
    uint64_t itl_samples = 0;
    double cumulative_ttft_ms = 0.0;
    double cumulative_itl_ms = 0.0;
    double cumulative_step_ms = 0.0;
    uint64_t packed_decode_steps = 0;
    uint64_t packed_decode_tokens = 0;
    uint64_t lane_decode_tokens = 0;
    uint64_t packed_fallback_batches = 0;
    uint64_t maximum_packed_batch = 0;
    uint64_t segmented_paged_decode_steps = 0;
    uint64_t segmented_paged_decode_tokens = 0;
    double cumulative_packed_decode_ms = 0.0;
    uint64_t prefix_cache_hits = 0;
    uint64_t prefix_cache_misses = 0;
    uint64_t prefix_cache_inserts = 0;
    uint64_t prefix_cache_evictions = 0;
    uint64_t prefix_cache_partial_hits = 0;
    uint64_t prefix_reused_tokens = 0;
    uint64_t prefix_cow_pages = 0;
    uint64_t direct_paged_prefill_tokens = 0;
    uint64_t ragged_prefill_steps = 0;
    uint64_t ragged_prefill_tokens = 0;
    uint64_t lane_prefill_tokens = 0;
    uint64_t maximum_ragged_prefill_batch = 0;
    double cumulative_ragged_prefill_ms = 0.0;
    uint64_t prefix_radix_nodes = 0;
    uint64_t prefix_radix_lookups = 0;
    uint64_t prefix_cow_bytes_copied = 0;
    uint64_t prefix_cow_bytes_saved = 0;
    uint64_t physical_kv_bytes = 0;

    double average_ttft_ms() const {
        return ttft_samples ? cumulative_ttft_ms / ttft_samples : 0.0;
    }
    double average_itl_ms() const {
        return itl_samples ? cumulative_itl_ms / itl_samples : 0.0;
    }
};

struct RequestLifecycleMetrics {
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t cancelled = 0;
    uint64_t failed = 0;
    uint64_t active = 0;
    uint64_t queued = 0;
    double average_ttft_ms = 0.0;
    double average_itl_ms = 0.0;
};

struct SchedulerMetrics {
    uint64_t steps = 0;
    double cumulative_step_ms = 0.0;
};

struct PrefillMetrics {
    uint64_t tokens = 0;
    uint64_t direct_paged_tokens = 0;
    uint64_t ragged_steps = 0;
    uint64_t ragged_tokens = 0;
    uint64_t lane_tokens = 0;
    uint64_t maximum_ragged_batch = 0;
    double cumulative_ragged_ms = 0.0;
};

struct DecodeMetrics {
    uint64_t tokens = 0;
    uint64_t packed_steps = 0;
    uint64_t packed_tokens = 0;
    uint64_t lane_tokens = 0;
    uint64_t fallback_batches = 0;
    uint64_t maximum_packed_batch = 0;
    uint64_t segmented_steps = 0;
    uint64_t segmented_tokens = 0;
    double cumulative_packed_ms = 0.0;
};

struct PrefixRuntimeMetrics {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t inserts = 0;
    uint64_t evictions = 0;
    uint64_t partial_hits = 0;
    uint64_t reused_tokens = 0;
    uint64_t cow_pages = 0;
    uint64_t radix_nodes = 0;
    uint64_t radix_lookups = 0;
    uint64_t cow_bytes_copied = 0;
    uint64_t cow_bytes_saved = 0;
};

struct KvMemoryMetrics {
    uint64_t pages_used = 0;
    uint64_t pages_total = 0;
    uint64_t bytes = 0;
};

struct GroupedConcurrentMetrics {
    RequestLifecycleMetrics requests;
    SchedulerMetrics scheduler;
    PrefillMetrics prefill;
    DecodeMetrics decode;
    PrefixRuntimeMetrics prefix;
    KvMemoryMetrics kv;
};

GroupedConcurrentMetrics group_concurrent_metrics(
    const ConcurrentMetrics& flat);

}
