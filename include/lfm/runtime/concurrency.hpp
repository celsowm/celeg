#pragma once

#include "lfm/model/execution/runtime_types.hpp"
#include "lfm/runtime/concurrency/policy.hpp"
#include "lfm/runtime/concurrency/metrics.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

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

struct ConcurrentRequestOptions {
    int max_new_tokens = 128;
    int eos_token = 7;
    int priority = 0;
    GenerationConfig generation{};
};

struct PollResult {
    RequestStatus status = RequestStatus::Queued;
    std::vector<int32_t> tokens;
    bool finished = false;
    std::string error;
};

// Stable serving facade. Scheduling, request ownership, worker lifecycle and
// CUDA execution live behind Impl and can evolve independently of callers.
class ConcurrentEngine {
public:
    using RequestId = uint64_t;

    ConcurrentEngine(std::string model_path,
                     int max_context,
                     ModelOptions model_options = {},
                     ConcurrentEngineOptions engine_options = {});
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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lfm
