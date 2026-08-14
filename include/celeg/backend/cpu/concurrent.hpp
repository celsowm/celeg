#pragma once

#include "celeg/backend/cpu/model.hpp"
#include "celeg/runtime/concurrency/metrics.hpp"
#include "celeg/runtime/concurrency/policy.hpp"
#include "celeg/runtime/request_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

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

// CPU-only counters with no backend-neutral meaning yet (no CUDA equivalent
// exists for chunked-prefill step accounting or CPU attention parallelism).
// These live alongside the neutral `ConcurrentMetrics` snapshot instead of
// duplicating it; see docs/SOLID_REVIEW_BACKENDS.md §4.1.
struct CpuConcurrentMetricsExtras {
    uint64_t chunked_prefill_steps = 0;
    uint64_t maximum_prefill_chunk = 0;
    uint64_t attention_parallel_calls = 0;
    double cumulative_direct_prefill_ms = 0.0;
};

struct CpuSchedulerDriver;

class CpuConcurrentEngine {
public:
    CpuConcurrentEngine(const std::string& model_path,
                        int max_context,
                        CpuModelOptions model_options = {},
                        CpuConcurrentEngineOptions engine_options = {},
                        std::shared_ptr<const RuntimeContext> runtime = nullptr);
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

    ConcurrentMetrics metrics() const;
    CpuConcurrentMetricsExtras metrics_extras() const;
    std::string backend_description() const;
    std::string last_error() const;

private:
    std::unique_ptr<CpuSchedulerDriver> state_;
};

} // namespace celeg
