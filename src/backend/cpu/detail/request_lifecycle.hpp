#pragma once

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/runtime/concurrency/metrics.hpp"
#include "concurrent_request.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

class CpuRequestLifecycle {
public:
    using Request = CpuConcurrentRequest;

    static void fail(const std::shared_ptr<Request>& request,
                     const std::string& error,
                     ConcurrentMetrics& metrics);
    static void cancel(const std::shared_ptr<Request>& request,
                       ConcurrentMetrics& metrics);
    static void observe_attention(const std::shared_ptr<Request>& request,
                                  CpuConcurrentMetricsExtras& extras);
    static void apply_decode_token(
        const std::shared_ptr<Request>& request,
        int32_t token,
        std::chrono::steady_clock::time_point now,
        ConcurrentMetrics& metrics,
        CpuConcurrentMetricsExtras& extras);
    static void apply_prefill(
        const std::shared_ptr<Request>& request,
        size_t consumed_tokens,
        ConcurrentMetrics& metrics);
};

}
