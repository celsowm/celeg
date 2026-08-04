#pragma once

#include "celeg/model/runtime_types.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/runtime/concurrency/policy.hpp"
#include "celeg/runtime/concurrency/metrics.hpp"
#include "celeg/runtime/request_types.hpp"
#include "celeg/runtime/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

struct ConcurrentEngineOptions {
    int max_active_requests = 8;
    int max_batched_tokens = 512;
    int prefill_chunk_tokens = 256;
    int page_tokens = 16;
    // Retained field name for C/C++ API compatibility; these are physical GPU pages.
    size_t logical_kv_pages = 0; // 0 = derive from max_active_requests * max_context.
    SchedulerPolicy scheduler_policy = SchedulerPolicy::GuaranteedNoEvict;
    bool worker_thread = true;
    int idle_sleep_microseconds = 100;
    bool packed_decode = true;
    int packed_min_batch = 1;
    bool ragged_packed_prefill = true;
    int ragged_prefill_min_batch = 2;
    bool prefix_cache = true;
    size_t prefix_cache_entries = 64;
};

struct CudaSchedulerDriver;

// Stable serving facade. Scheduling, request ownership, worker lifecycle and
// CUDA execution live behind CudaSchedulerDriver and can evolve independently
// of callers.
class ConcurrentEngine {
public:
    using RequestId = celeg::RequestId;

    ConcurrentEngine(std::string model_path,
                     int max_context,
                     CudaModelOptions model_options = {},
                     ConcurrentEngineOptions engine_options = {},
                     std::shared_ptr<const RuntimeContext> runtime = nullptr);
    ~ConcurrentEngine();

    ConcurrentEngine(const ConcurrentEngine&) = delete;
    ConcurrentEngine& operator=(const ConcurrentEngine&) = delete;
    ConcurrentEngine(ConcurrentEngine&&) = delete;
    ConcurrentEngine& operator=(ConcurrentEngine&&) = delete;

    RequestId submit(std::vector<int32_t> prompt,
                     ConcurrentRequestOptions options = {});
    bool cancel(RequestId id);
    PollResult poll(RequestId id, size_t max_tokens = 0);
    RequestStatus status(RequestId id) const;

    // Frees the request record. Only valid once the request has reached a
    // terminal status; returns false otherwise or if the id is unknown.
    bool release(RequestId id);

    // Executes one scheduler iteration. Useful for embedding the runtime in an
    // external event loop. Returns true when useful work was performed.
    bool step();
    void start();
    void stop();
    bool running() const;

    ConcurrentMetrics metrics() const;
    GroupedConcurrentMetrics grouped_metrics() const;

private:
    std::unique_ptr<CudaSchedulerDriver> state_;
};

} // namespace celeg
