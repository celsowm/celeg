#include "backend/cuda/concurrency.hpp"

#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/model.hpp"
#include "celeg/model/resolved.hpp"
#include "packed/executor.hpp"
#include "backend/cuda/paged_kv.hpp"
#include "kernels/mmq.hpp"
#include "celeg/runtime/cache/prefix_cache.hpp"
#include "runtime/concurrency/detail/request_registry.hpp"
#include "runtime/concurrency/detail/batch_planner.hpp"
#include "runtime/concurrency/detail/worker.hpp"
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

CudaSchedulerDriver::CudaSchedulerDriver(std::string model_path,
                                   int max_context,
                                   CudaModelOptions model_options,
                                   ConcurrentEngineOptions engine_options,
                                   std::shared_ptr<const RuntimeContext> runtime)
    : model_path_(std::move(model_path)),
      max_context_(max_context),
      model_options_(model_options),
      engine_options_(engine_options),
      runtime_(runtime ? std::move(runtime) : create_builtin_runtime_context()),
      weight_cache_(std::make_shared<CudaWeightCache>()) {
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
    const detail::ModelBootstrap bootstrap = detail::load_model_bootstrap(
        std::filesystem::path(model_path_), *runtime_);
    topology_ = bootstrap.model.topology;
    program_ = build_model_program(bootstrap.model);
    const int required_initial_span = program_.required_initial_attention_span();
    if (required_initial_span > max_context_) {
        throw std::invalid_argument(
            "attention prefix length exceeds max_context");
    }
    if (required_initial_span > 0 && !engine_options_.packed_decode) {
        throw std::invalid_argument(
            "CUDA concurrent Prefix-LM execution requires packed_decode");
    }
    if (required_initial_span > engine_options_.max_batched_tokens) {
        throw std::invalid_argument(
            "CUDA Prefix-LM prefix exceeds max_batched_tokens capacity");
    }
    if (engine_options_.packed_decode) {
        paged_kv_ = std::make_unique<PhysicalPagedKvCache>(
            total_pages, engine_options_.page_tokens, max_context_,
            model_options_.kv_cache_mode, topology_.exec, program_);
    }
    if (paged_kv_) {
        prefix_cache_ = std::make_unique<PrefixCacheManager>(
            *paged_kv_, engine_options_.prefix_cache,
            engine_options_.prefix_cache_entries);
    }
    lanes_.reserve(active);
    for (size_t i = 0; i < active; ++i) {
        auto lane = std::make_unique<Lane>();
        lane->index = static_cast<int>(i);
        lanes_.push_back(std::move(lane));
    }
    metrics_.logical_pages_total = paged_kv_ ? total_pages : 0;
    metrics_.physical_kv_bytes = paged_kv_ ? paged_kv_->memory_bytes() : 0;
    if (engine_options_.packed_decode) {
        packed_decode_output_.resize(active);
        CudaModelOptions packed_options = model_options_;
        packed_options.allocate_local_kv_cache = false;
        packed_executor_ = std::make_unique<PackedDecodeExecutor>(
            static_cast<size_t>(engine_options_.max_active_requests),
            static_cast<size_t>(engine_options_.max_batched_tokens),
            paged_kv_.get(), topology_.exec, program_, topology_.dims.vocab_size,
            CudaExecutionPlan::compile(
                packed_options, max_context_, discover_cuda_device_capabilities()));
    }
    if (engine_options_.worker_thread) start();
}

}