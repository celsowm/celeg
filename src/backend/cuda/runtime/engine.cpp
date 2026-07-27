#include "lfm/runtime/concurrency.hpp"

#include "lfm/model/config/config.hpp"
#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/model.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/backend/cuda/packed.hpp"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/runtime/cache/prefix_cache.hpp"
#include "lfm/detail/runtime/concurrency/request_registry.hpp"
#include "lfm/detail/runtime/concurrency/batch_planner.hpp"
#include "lfm/detail/runtime/concurrency/worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfm {

struct ConcurrentEngine::Impl {
    using RequestId = ConcurrentEngine::RequestId;
    using Request = detail::RequestRecord;

    struct Lane {
        int index = -1;
        std::unique_ptr<LfmModel> model;
        RequestId request_id = 0;
    };

    Impl(std::string safetensors_path, int max_context,
         ModelOptions model_options, ConcurrentEngineOptions engine_options);
    ~Impl();

    RequestId submit(std::vector<int32_t> prompt, ConcurrentRequestOptions options);
    bool cancel(RequestId id);
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

    std::string safetensors_path_;
    int max_context_;
    ModelOptions model_options_;
    ConcurrentEngineOptions engine_options_;
    ModelShape shape_;

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

ConcurrentEngine::Impl::Impl(std::string safetensors_path,
                                   int max_context,
                                   ModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : safetensors_path_(std::move(safetensors_path)),
      max_context_(max_context),
      model_options_(model_options),
      engine_options_(engine_options) {
    if (max_context_ <= 0) throw std::invalid_argument("max_context must be positive");
    if (engine_options_.max_active_requests <= 0)
        throw std::invalid_argument("max_active_requests must be positive");
    if (engine_options_.max_batched_tokens <= 0)
        throw std::invalid_argument("max_batched_tokens must be positive");
    if (engine_options_.prefill_chunk_tokens <= 0)
        throw std::invalid_argument("prefill_chunk_tokens must be positive");
    if (engine_options_.page_tokens <= 0)
        throw std::invalid_argument("page_tokens must be positive");
    if (engine_options_.packed_min_batch < 1)
        throw std::invalid_argument("packed_min_batch must be positive");
    if (engine_options_.ragged_prefill_min_batch < 1)
        throw std::invalid_argument("ragged_prefill_min_batch must be positive");
    if (engine_options_.prefix_cache_entries == 0 && engine_options_.prefix_cache)
        throw std::invalid_argument("prefix_cache_entries must be positive when prefix cache is enabled");

    const size_t pages_per_lane =
        (static_cast<size_t>(max_context_) + engine_options_.page_tokens - 1) /
        static_cast<size_t>(engine_options_.page_tokens);
    const size_t active = static_cast<size_t>(engine_options_.max_active_requests);
    if (engine_options_.logical_kv_pages == 0 &&
        pages_per_lane > std::numeric_limits<size_t>::max() / active) {
        throw std::overflow_error("derived physical KV page count overflows size_t");
    }
    const size_t total_pages = engine_options_.logical_kv_pages != 0
        ? engine_options_.logical_kv_pages
        : pages_per_lane * active;
    // Load the model topology so the physical paged KV arena and the packed
    // executor can size per-attention-layer storage from the variant shape.
    shape_ = detail::load_model_bootstrap(
        std::filesystem::path(safetensors_path_)).shape;
    paged_kv_ = std::make_unique<PhysicalPagedKvCache>(
        total_pages, engine_options_.page_tokens, max_context_,
        model_options_.kv_cache_mode, shape_);
    prefix_cache_ = std::make_unique<PrefixCacheManager>(
        *paged_kv_, engine_options_.prefix_cache,
        engine_options_.prefix_cache_entries);
    lanes_.reserve(static_cast<size_t>(engine_options_.max_active_requests));
    for (int i = 0; i < engine_options_.max_active_requests; ++i) {
        auto lane = std::make_unique<Lane>();
        lane->index = i;
        lanes_.push_back(std::move(lane));
    }
    metrics_.logical_pages_total = total_pages;
    metrics_.physical_kv_bytes = paged_kv_->memory_bytes();
    if (engine_options_.packed_decode) {
        packed_executor_ = std::make_unique<PackedDecodeExecutor>(
            static_cast<size_t>(engine_options_.max_active_requests),
            static_cast<size_t>(engine_options_.max_batched_tokens),
            paged_kv_.get(), shape_);
    }
    if (engine_options_.worker_thread) start();
}

#include "request_lifecycle.inl"
#include "prefill_scheduler.inl"
#include "decode_scheduler.inl"
#include "engine_runtime.inl"

} // namespace lfm
