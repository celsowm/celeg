#pragma once

#include "lfm/backend/cpu/model.hpp"
#include "lfm/runtime/concurrency.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

struct CpuConcurrentEngineOptions {
    size_t max_active_requests = 16;
    size_t max_batched_tokens = 256;
    size_t max_prefill_batch = 16;
    size_t max_decode_batch = 16;
    size_t long_prefill_chunk_tokens = 256;
    size_t long_prefill_threshold = 32;
    bool decode_first = true;
    bool worker_thread = true;
    bool prefix_cache = true;
    size_t prefix_cache_max_entries = 256;
    size_t prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
};

struct CpuConcurrentMetrics {
    uint64_t submitted_requests = 0;
    uint64_t completed_requests = 0;
    uint64_t cancelled_requests = 0;
    uint64_t failed_requests = 0;
    uint64_t active_requests = 0;
    uint64_t queued_requests = 0;

    uint64_t prefill_tokens = 0;
    uint64_t decode_tokens = 0;
    uint64_t ragged_prefill_steps = 0;
    uint64_t packed_decode_steps = 0;
    uint64_t chunked_prefill_steps = 0;
    uint64_t chunked_prefill_tokens = 0;
    uint64_t maximum_prefill_batch = 0;
    uint64_t maximum_decode_batch = 0;
    uint64_t maximum_prefill_chunk = 0;
    uint64_t prefix_cache_hits = 0;
    uint64_t prefix_cache_misses = 0;
    uint64_t prefix_cache_partial_hits = 0;
    uint64_t prefix_cache_inserts = 0;
    uint64_t prefix_cache_evictions = 0;
    uint64_t prefix_reused_tokens = 0;
    uint64_t prefix_cow_pages = 0;
    uint64_t prefix_cow_bytes = 0;
    uint64_t attention_parallel_calls = 0;

    double cumulative_prefill_ms = 0.0;
    double cumulative_decode_ms = 0.0;
    double cumulative_scheduler_ms = 0.0;
    double cumulative_ttft_ms = 0.0;
    uint64_t ttft_samples = 0;
    double cumulative_itl_ms = 0.0;
    uint64_t itl_samples = 0;

    double prefill_tokens_per_second() const;
    double decode_tokens_per_second() const;
    double average_ttft_ms() const;
    double average_itl_ms() const;
};

class CpuConcurrentEngine {
public:
    CpuConcurrentEngine(const std::string& model_path,
                        int max_context,
                        CpuModelOptions model_options = {},
                        CpuConcurrentEngineOptions engine_options = {});
    ~CpuConcurrentEngine();

    CpuConcurrentEngine(const CpuConcurrentEngine&) = delete;
    CpuConcurrentEngine& operator=(const CpuConcurrentEngine&) = delete;

    RequestId submit(std::vector<int32_t> prompt,
                     ConcurrentRequestOptions options = {});
    PollResult poll(RequestId request_id, size_t max_tokens = 0);
    RequestStatus status(RequestId request_id) const;
    bool cancel(RequestId request_id);

    // Frees the request record. Only valid once the request has reached a
    // terminal status; returns false otherwise or if the id is unknown.
    bool release(RequestId request_id);

    // Runs one scheduling iteration. Returns true when any request advanced.
    bool step();
    void start();
    void stop();

    CpuConcurrentMetrics metrics() const;
    std::string backend_description() const;
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lfm
