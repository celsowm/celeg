#include "celeg/runtime/concurrency.hpp"

#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/model/model.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/runtime/cache/prefix_cache.hpp"
#include "celeg/detail/runtime/concurrency/request_registry.hpp"
#include "celeg/detail/runtime/concurrency/batch_planner.hpp"
#include "celeg/detail/runtime/concurrency/worker.hpp"
#include "engine_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

ConcurrentEngine::Impl::Impl(std::string model_path,
                                   int max_context,
                                   ModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : model_path_(std::move(model_path)),
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
    // executor can size per-attention-layer storage from the resolved topology.
    shape_ = detail::load_model_bootstrap(
        std::filesystem::path(model_path_)).model.topology;
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

} // namespace celeg
