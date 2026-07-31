#pragma once

#include "celeg/runtime/concurrency.hpp"
#include "celeg/model/model.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/runtime/cache/prefix_cache.hpp"
#include "celeg/detail/runtime/concurrency/request_registry.hpp"
#include "celeg/detail/runtime/concurrency/batch_planner.hpp"
#include "celeg/detail/runtime/concurrency/worker.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace celeg {

struct ConcurrentEngine::Impl {
    using RequestId = ConcurrentEngine::RequestId;
    using Request = detail::RequestRecord;

    struct Lane {
        int index = -1;
        std::unique_ptr<Model> model;
        RequestId request_id = 0;
    };

    Impl(std::string model_path, int max_context,
         ModelOptions model_options, ConcurrentEngineOptions engine_options);
    ~Impl();

    RequestId submit(std::vector<int32_t> prompt, ConcurrentRequestOptions options);
    bool cancel(RequestId id);
    bool release(RequestId id);
    PollResult poll(RequestId id, size_t max_tokens);
    RequestStatus status(RequestId id) const;
    bool step();
    void start();
    void stop();
    bool running() const;
    ConcurrentMetrics metrics() const;
    GroupedConcurrentMetrics grouped_metrics() const;

private:
    bool admit_requests_locked();
    bool run_prefill_work();
    bool run_decode_work();
    void finish_request_locked(Request& request, RequestStatus status,
                               std::string error = {});
    void complete_prefill_locked(Request& request, Lane& lane);
    Lane* find_free_lane_locked();
    std::vector<detail::LaneSnapshot> lane_snapshots_locked() const;

    std::string model_path_;
    int max_context_;
    ModelOptions model_options_;
    ConcurrentEngineOptions engine_options_;
    RuntimeTopology shape_;

    mutable std::mutex mutex_;
    std::mutex step_mutex_;
    detail::RequestRegistry registry_;
    detail::BatchPlanner planner_;
    detail::EngineWorker worker_;
    std::vector<std::unique_ptr<Lane>> lanes_;
    std::unique_ptr<PhysicalPagedKvCache> paged_kv_;
    std::unique_ptr<PrefixCacheManager> prefix_cache_;
    std::unique_ptr<PackedDecodeExecutor> packed_executor_;
    ConcurrentMetrics metrics_;
    bool stopping_ = false;
};

} // namespace celeg
