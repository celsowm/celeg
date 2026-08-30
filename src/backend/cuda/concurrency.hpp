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

struct ConcurrentEngineOptions : ConcurrentSchedulerOptions {
    ConcurrentEngineOptions()
        : ConcurrentSchedulerOptions(8, 512, true, true) {}

    int prefill_chunk_tokens = 256;
    int page_tokens = 16;
    size_t logical_kv_pages = 0;
    SchedulerPolicy scheduler_policy = SchedulerPolicy::GuaranteedNoEvict;
    int idle_sleep_microseconds = 100;
    bool packed_decode = true;
    int packed_min_batch = 1;
    bool ragged_packed_prefill = true;
    int ragged_prefill_min_batch = 2;
    size_t prefix_cache_entries = 64;
};

struct CudaSchedulerDriver;

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

    bool release(RequestId id);

    bool step();
    void start();
    void stop();
    bool running() const;

    ConcurrentMetrics metrics() const;
    GroupedConcurrentMetrics grouped_metrics() const;

private:
    std::unique_ptr<CudaSchedulerDriver> state_;
};

}
