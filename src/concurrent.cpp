#include "lfm/concurrent.hpp"

#include "lfm/config.hpp"
#include "lfm/model.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/model_variant.hpp"
#include "lfm/packed.hpp"
#include "lfm/paged_kv.hpp"
#include "lfm/prefix_cache.hpp"
#include "lfm/detail/request_registry.hpp"
#include "lfm/detail/batch_planner.hpp"
#include "lfm/detail/engine_worker.hpp"

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
    const std::filesystem::path config_path =
        std::filesystem::path(safetensors_path_).parent_path() / "config.json";
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error(
            "config.json not found alongside checkpoint: " + config_path.string());
    }
    ModelConfig config = ModelConfig::load(config_path.string());
    shape_ = ModelShape::from_config(config);
    {
        const IModelVariant& variant = ModelVariantRegistry::instance().select(shape_);
        shape_ = variant.resolve_shape(shape_);
    }
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
            paged_kv_.get(), shape_);
    }
    if (engine_options_.worker_thread) start();
}

ConcurrentEngine::Impl::~Impl() {
    stop();
    std::lock_guard<std::mutex> lock(mutex_);
    if (prefix_cache_) prefix_cache_->clear();
}

ConcurrentEngine::RequestId ConcurrentEngine::Impl::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    if (prompt.empty()) throw std::invalid_argument("request prompt is empty");
    if (prompt.size() > static_cast<size_t>(max_context_))
        throw std::invalid_argument("request prompt exceeds max_context");
    if (options.max_new_tokens <= 0)
        throw std::invalid_argument("max_new_tokens must be positive");
    if (prompt.size() + static_cast<size_t>(options.max_new_tokens) >
        static_cast<size_t>(max_context_)) {
        throw std::invalid_argument("prompt plus max_new_tokens exceeds max_context");
    }
    options.generation.validate();

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) throw std::runtime_error("engine is stopping");
    const RequestId id = registry_.create(
        std::move(prompt), std::move(options),
        std::chrono::steady_clock::now());
    ++metrics_.submitted;
    metrics_.queued_requests = registry_.queued_count();
    worker_.notify();
    return id;
}

bool ConcurrentEngine::Impl::cancel(RequestId id) {
    std::lock_guard<std::mutex> lock(mutex_);
    Request* found = registry_.find(id);
    if (!found) return false;
    Request& request = *found;
    if (request.status == RequestStatus::Finished ||
        request.status == RequestStatus::Cancelled ||
        request.status == RequestStatus::Failed) return false;
    request.cancel_requested = true;
    worker_.notify();
    return true;
}

PollResult ConcurrentEngine::Impl::poll(RequestId id, size_t max_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    Request& request = registry_.at(id);
    PollResult result;
    result.status = request.status;
    result.finished = request.status == RequestStatus::Finished ||
                      request.status == RequestStatus::Cancelled ||
                      request.status == RequestStatus::Failed;
    result.error = request.error;
    const size_t count = max_tokens == 0
        ? request.output.size()
        : std::min(max_tokens, request.output.size());
    result.tokens.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.tokens.push_back(request.output.front());
        request.output.pop_front();
    }
    return result;
}

RequestStatus ConcurrentEngine::Impl::status(RequestId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registry_.at(id).status;
}

ConcurrentEngine::Impl::Lane* ConcurrentEngine::Impl::find_free_lane_locked() {
    for (auto& lane : lanes_) if (lane->request_id == 0) return lane.get();
    return nullptr;
}

std::vector<detail::LaneSnapshot>
ConcurrentEngine::Impl::lane_snapshots_locked() const {
    std::vector<detail::LaneSnapshot> result;
    result.reserve(lanes_.size());
    for (const auto& lane : lanes_) {
        detail::LaneSnapshot snapshot;
        snapshot.lane_index = lane->index;
        snapshot.request_id = lane->request_id;
        if (lane->request_id != 0) {
            const Request& request = registry_.at(lane->request_id);
            snapshot.status = request.status;
            snapshot.priority = request.options.priority;
        }
        result.push_back(snapshot);
    }
    return result;
}

bool ConcurrentEngine::Impl::admit_requests_locked() {
    bool did_work = false;
    while (!registry_.empty_queue()) {
        Lane* lane = find_free_lane_locked();
        if (!lane) break;
        const auto selected = planner_.next_admission(registry_);
        if (!selected) break;
        const RequestId id = *selected;
        Request& request = registry_.at(id);
        if (request.cancel_requested) {
            registry_.erase_queued(id);
            finish_request_locked(request, RequestStatus::Cancelled);
            did_work = true;
            continue;
        }

        const size_t reserved_tokens =
            engine_options_.scheduler_policy == SchedulerPolicy::GuaranteedNoEvict
                ? request.prompt.size() + static_cast<size_t>(request.options.max_new_tokens)
                : request.prompt.size() + static_cast<size_t>(engine_options_.page_tokens);
        PrefixAcquireResult prefix = prefix_cache_->acquire(
            request.prompt, reserved_tokens);
        if (prefix.status == PrefixAcquireStatus::OutOfMemory) break;

        const bool prefix_hit = prefix.status == PrefixAcquireStatus::Hit;
        std::vector<uint32_t> pages;
        if (prefix_hit) {
            pages = std::move(prefix.pages);
        } else {
            auto allocated = prefix_cache_->allocate_request_pages(reserved_tokens);
            if (!allocated) break;
            pages = std::move(*allocated);
        }

        try {
            if (!lane->model) {
                ModelOptions lane_options = model_options_;
                lane_options.allocate_local_kv_cache =
                    !engine_options_.packed_decode;
                lane->model = std::make_unique<LfmModel>(
                    safetensors_path_, max_context_, lane_options,
                    request.options.generation);
            } else {
                lane->model->set_generation_config(request.options.generation);
            }
        } catch (const std::exception& error) {
            paged_kv_->release(pages);
            registry_.erase_queued(id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
            did_work = true;
            continue;
        }

        registry_.erase_queued(id);
        request.pages = std::move(pages);
        request.lane_index = lane->index;
        lane->request_id = id;
        try {
            lane->model->release_local_kv_cache();
            lane->model->reset(false);
            if (prefix_hit) {
                if (!prefix.state) {
                    throw std::runtime_error("prefix cache hit has no session state");
                }
                lane->model->restore_prefix_state(*prefix.state);
                request.prefill_offset = prefix.matched_tokens;
                request.status = request.prefill_offset == request.prompt.size()
                    ? RequestStatus::Decoding : RequestStatus::Prefill;
                request.paged_ready = true;
            } else {
                request.status = RequestStatus::Prefill;
                request.paged_ready = packed_executor_ != nullptr;
            }
        } catch (const std::exception& error) {
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
        did_work = true;
    }
    metrics_.queued_requests = registry_.queued_count();
    metrics_.active_requests = 0;
    for (const auto& lane : lanes_) {
        if (lane->request_id != 0) ++metrics_.active_requests;
    }
    metrics_.logical_pages_used = paged_kv_->used_pages();
    return did_work;
}

void ConcurrentEngine::Impl::finish_request_locked(Request& request,
                                             RequestStatus status_value,
                                             std::string error) {
    if (request.lane_index >= 0) {
        Lane& lane = *lanes_.at(static_cast<size_t>(request.lane_index));
        lane.request_id = 0;
        request.lane_index = -1;
    }
    if (!request.pages.empty()) {
        paged_kv_->release(request.pages);
        request.pages.clear();
    }
    request.status = status_value;
    request.error = std::move(error);
    if (status_value == RequestStatus::Finished) ++metrics_.completed;
    if (status_value == RequestStatus::Cancelled) ++metrics_.cancelled;
    if (status_value == RequestStatus::Failed) ++metrics_.failed;
    metrics_.logical_pages_used = paged_kv_->used_pages();
}


void ConcurrentEngine::Impl::complete_prefill_locked(Request& request, Lane& lane) {
    request.paged_ready = packed_executor_ != nullptr;
    if (packed_executor_ && prefix_cache_->enabled()) {
        PrefixState state = lane.model->export_prefix_state();
        (void)prefix_cache_->insert_or_update(
            request.prompt, request.pages, std::move(state));
    }
    request.status = RequestStatus::Decoding;
}

bool ConcurrentEngine::Impl::run_prefill_work() {
    struct Work { Lane* lane; RequestId id; size_t begin; size_t count; bool first; bool final; };
    std::vector<Work> work;
    int token_budget = engine_options_.max_batched_tokens;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int decode_reservation = 0;
        for (const auto& lane_ptr : lanes_) {
            if (lane_ptr->request_id == 0) continue;
            const Request& request = registry_.at(lane_ptr->request_id);
            if (request.status == RequestStatus::Decoding &&
                !request.cancel_requested) ++decode_reservation;
        }
        token_budget = std::max(0, token_budget - decode_reservation);
        const std::vector<int> ordered =
            planner_.order_prefill(lane_snapshots_locked());
        for (const int lane_index : ordered) {
            auto& lane_ptr = lanes_.at(static_cast<size_t>(lane_index));
            if (token_budget <= 0) break;
            Lane& lane = *lane_ptr;
            if (lane.request_id == 0) continue;
            Request& request = registry_.at(lane.request_id);
            if (request.status != RequestStatus::Prefill) continue;
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            const size_t remaining = request.prompt.size() - request.prefill_offset;
            const size_t count = std::min<size_t>(
                remaining, std::min(engine_options_.prefill_chunk_tokens,
                                    token_budget));
            if (count == 0) continue;
            work.push_back({&lane, request.id, request.prefill_offset, count,
                            request.prefill_offset == 0,
                            count == remaining});
            token_budget -= static_cast<int>(count);
        }
    }
    if (packed_executor_ && engine_options_.ragged_packed_prefill &&
        work.size() >= static_cast<size_t>(engine_options_.ragged_prefill_min_batch)) {
        size_t maximum_tokens = 0;
        for (const Work& item : work) maximum_tokens = std::max(maximum_tokens, item.count);
        for (size_t wave = 0; wave < maximum_tokens; ++wave) {
            std::vector<Work> active;
            std::vector<LfmModel*> models;
            std::vector<std::vector<uint32_t>> page_tables;
            std::vector<int32_t> tokens;
            std::vector<uint8_t> finalize_rows;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const Work& item : work) {
                    if (wave >= item.count) continue;
                    Request& request = registry_.at(item.id);
                    if (request.status != RequestStatus::Prefill ||
                        request.cancel_requested) continue;
                    active.push_back(item);
                    models.push_back(item.lane->model.get());
                    page_tables.push_back(request.pages);
                    tokens.push_back(request.prompt[item.begin + wave]);
                    finalize_rows.push_back(
                        static_cast<uint8_t>(item.final && wave + 1 == item.count));
                }
            }
            if (active.empty()) continue;
            try {
                const PackedDecodeMetrics before = packed_executor_->metrics();
                packed_executor_->prefill_step(models, page_tables, tokens,
                                               finalize_rows);
                const PackedDecodeMetrics after = packed_executor_->metrics();
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_.ragged_prefill_steps +=
                    after.ragged_prefill_steps - before.ragged_prefill_steps;
                metrics_.cumulative_ragged_prefill_ms +=
                    after.cumulative_prefill_ms - before.cumulative_prefill_ms;
                metrics_.maximum_ragged_prefill_batch = std::max<uint64_t>(
                    metrics_.maximum_ragged_prefill_batch,
                    static_cast<uint64_t>(active.size()));
                for (size_t row = 0; row < active.size(); ++row) {
                    const Work& item = active[row];
                    Request& request = registry_.at(item.id);
                    if (request.cancel_requested) {
                        finish_request_locked(request, RequestStatus::Cancelled);
                        continue;
                    }
                    if (request.status != RequestStatus::Prefill) continue;
                    ++request.prefill_offset;
                    ++metrics_.prefill_tokens;
                    ++metrics_.direct_paged_prefill_tokens;
                    ++metrics_.ragged_prefill_tokens;
                    if (finalize_rows[row] != 0) {
                        complete_prefill_locked(request, *item.lane);
                    }
                }
            } catch (const std::exception& error) {
                std::lock_guard<std::mutex> lock(mutex_);
                for (const Work& item : active) {
                    Request& request = registry_.at(item.id);
                    if (request.status == RequestStatus::Prefill) {
                        finish_request_locked(request, RequestStatus::Failed,
                                              error.what());
                    }
                }
                break;
            }
        }
        return !work.empty();
    }
    for (const Work& item : work) {
        try {
            std::vector<int32_t> chunk;
            std::vector<uint32_t> page_table;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                Request& request = registry_.at(item.id);
                chunk.assign(request.prompt.begin() + static_cast<ptrdiff_t>(item.begin),
                             request.prompt.begin() + static_cast<ptrdiff_t>(item.begin + item.count));
                page_table = request.pages;
            }
            if (packed_executor_) {
                item.lane->model->prefill_chunk_paged(
                    chunk, item.first, item.final, *paged_kv_, page_table);
            } else if (item.first && item.final) {
                item.lane->model->prefill(chunk);
            } else {
                item.lane->model->prefill_chunk(chunk, item.first, item.final);
            }
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            request.prefill_offset += item.count;
            metrics_.prefill_tokens += item.count;
            metrics_.lane_prefill_tokens += item.count;
            if (packed_executor_) metrics_.direct_paged_prefill_tokens += item.count;
            if (item.final) {
                complete_prefill_locked(request, *item.lane);
            }
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    return !work.empty();
}

bool ConcurrentEngine::Impl::run_decode_work() {
    struct Work { Lane* lane; RequestId id; };
    std::vector<Work> work;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int decode_budget = engine_options_.max_batched_tokens;
        const std::vector<int> ordered =
            planner_.order_decode(lane_snapshots_locked());
        for (const int lane_index : ordered) {
            if (decode_budget <= 0) break;
            Lane& lane = *lanes_.at(static_cast<size_t>(lane_index));
            if (lane.request_id == 0) continue;
            Request& request = registry_.at(lane.request_id);
            if (request.status != RequestStatus::Decoding) continue;
            if (request.cancel_requested) {
                finish_request_locked(request, RequestStatus::Cancelled);
                continue;
            }
            if (engine_options_.scheduler_policy ==
                SchedulerPolicy::MaxUtilization) {
                const size_t next_tokens = request.prompt.size() +
                    static_cast<size_t>(request.generated + 1);
                const size_t capacity = request.pages.size() *
                    static_cast<size_t>(engine_options_.page_tokens);
                if (next_tokens > capacity) {
                    auto page = prefix_cache_->allocate_request_pages(1);
                    if (!page) continue;
                    request.pages.insert(request.pages.end(),
                                         page->begin(), page->end());
                    metrics_.logical_pages_used = paged_kv_->used_pages();
                }
            }
            work.push_back({&lane, request.id});
            --decode_budget;
        }
    }

    auto accept_token = [&](const Work& item, int32_t token,
                            bool packed_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        Request& request = registry_.at(item.id);
        if (request.cancel_requested) {
            finish_request_locked(request, RequestStatus::Cancelled);
            return;
        }
        if (request.status != RequestStatus::Decoding) return;
        request.output.push_back(token);
        const auto token_time = std::chrono::steady_clock::now();
        if (request.generated == 0) {
            metrics_.cumulative_ttft_ms +=
                std::chrono::duration<double, std::milli>(
                    token_time - request.submitted_at).count();
            ++metrics_.ttft_samples;
        } else {
            metrics_.cumulative_itl_ms +=
                std::chrono::duration<double, std::milli>(
                    token_time - request.last_token_at).count();
            ++metrics_.itl_samples;
        }
        request.last_token_at = token_time;
        ++request.generated;
        ++metrics_.decoded_tokens;
        if (packed_path) ++metrics_.packed_decode_tokens;
        else ++metrics_.lane_decode_tokens;
        if (token == request.options.eos_token ||
            request.generated >= request.options.max_new_tokens) {
            finish_request_locked(request, RequestStatus::Finished);
        }
    };

    std::vector<Work> packed_work;
    std::vector<Work> lane_work;
    packed_work.reserve(work.size());
    lane_work.reserve(work.size());
    for (const Work& item : work) {
        std::string reason;
        bool paged_ready = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            paged_ready = registry_.at(item.id).paged_ready;
        }
        if (paged_ready && packed_executor_ &&
            packed_executor_->eligible(*item.lane->model, &reason)) {
            packed_work.push_back(item);
        } else {
            lane_work.push_back(item);
        }
    }
    if (packed_work.size() <
        static_cast<size_t>(engine_options_.packed_min_batch)) {
        bool can_fallback = true;
        for (const Work& item : packed_work) {
            if (!item.lane->model->local_kv_cache_available()) {
                can_fallback = false;
                break;
            }
        }
        if (can_fallback) {
            lane_work.insert(lane_work.end(), packed_work.begin(), packed_work.end());
            packed_work.clear();
        }
    }

    if (!packed_work.empty()) {
        std::vector<LfmModel*> models;
        std::vector<std::vector<uint32_t>> page_tables;
        models.reserve(packed_work.size());
        page_tables.reserve(packed_work.size());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const Work& item : packed_work) {
                models.push_back(item.lane->model.get());
                page_tables.push_back(registry_.at(item.id).pages);
            }
        }
        try {
            const PackedDecodeMetrics before = packed_executor_->metrics();
            const std::vector<int32_t> tokens =
                packed_executor_->decode(models, page_tables);
            const PackedDecodeMetrics after = packed_executor_->metrics();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                metrics_.packed_decode_steps += after.steps - before.steps;
                metrics_.cumulative_packed_decode_ms +=
                    after.cumulative_ms - before.cumulative_ms;
                metrics_.segmented_paged_decode_steps +=
                    after.segmented_paged_steps - before.segmented_paged_steps;
                metrics_.segmented_paged_decode_tokens +=
                    after.segmented_paged_tokens - before.segmented_paged_tokens;
                metrics_.maximum_packed_batch = std::max<uint64_t>(
                    metrics_.maximum_packed_batch,
                    static_cast<uint64_t>(packed_work.size()));
            }
            for (size_t i = 0; i < packed_work.size(); ++i) {
                accept_token(packed_work[i], tokens[i], true);
            }
        } catch (const std::invalid_argument& error) {
            // A paged request normally has no local KV after prefill import.
            // Only fall back when every row still owns a valid contiguous cache;
            // otherwise surface the page-table/packed error instead of invoking
            // the lane path with null cache pointers.
            bool can_fallback = true;
            for (const Work& item : packed_work) {
                if (!item.lane->model->local_kv_cache_available()) {
                    can_fallback = false;
                    break;
                }
            }
            if (can_fallback) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++metrics_.packed_fallback_batches;
                }
                lane_work.insert(lane_work.end(), packed_work.begin(), packed_work.end());
            } else {
                for (const Work& item : packed_work) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    Request& request = registry_.at(item.id);
                    finish_request_locked(request, RequestStatus::Failed,
                                          error.what());
                }
            }
        } catch (const std::exception& error) {
            for (const Work& item : packed_work) {
                std::lock_guard<std::mutex> lock(mutex_);
                Request& request = registry_.at(item.id);
                finish_request_locked(request, RequestStatus::Failed,
                                      error.what());
            }
        }
    }

    // Requests not eligible for the packed kernel continue through the v0.0.9
    // multi-stream lane path. All streams are enqueued before any is joined.
    std::vector<Work> launched;
    launched.reserve(lane_work.size());
    for (const Work& item : lane_work) {
        try {
            item.lane->model->decode_async_begin();
            launched.push_back(item);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    for (const Work& item : launched) {
        try {
            const int32_t token = item.lane->model->decode_async_finish();
            accept_token(item, token, false);
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex_);
            Request& request = registry_.at(item.id);
            finish_request_locked(request, RequestStatus::Failed, error.what());
        }
    }
    return !work.empty();
}

bool ConcurrentEngine::Impl::step() {
    std::lock_guard<std::mutex> step_lock(step_mutex_);
    const auto started = std::chrono::steady_clock::now();
    bool did_work = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        did_work |= admit_requests_locked();
    }
    did_work |= run_prefill_work();
    did_work |= run_decode_work();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        did_work |= admit_requests_locked();
        ++metrics_.scheduler_steps;
        const auto ended = std::chrono::steady_clock::now();
        metrics_.cumulative_step_ms +=
            std::chrono::duration<double, std::milli>(ended - started).count();
    }
    return did_work;
}

void ConcurrentEngine::Impl::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    worker_.start([this] { return step(); },
                  engine_options_.idle_sleep_microseconds);
}

void ConcurrentEngine::Impl::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    worker_.stop();
}

bool ConcurrentEngine::Impl::running() const {
    return worker_.running();
}

ConcurrentMetrics ConcurrentEngine::Impl::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ConcurrentMetrics snapshot = metrics_;
    if (prefix_cache_) {
        const PrefixCacheMetrics& cache = prefix_cache_->metrics();
        snapshot.prefix_cache_hits = cache.hits;
        snapshot.prefix_cache_misses = cache.misses;
        snapshot.prefix_cache_inserts = cache.inserts;
        snapshot.prefix_cache_evictions = cache.evictions;
        snapshot.prefix_cache_partial_hits = cache.partial_hits;
        snapshot.prefix_reused_tokens = cache.reused_tokens;
        snapshot.prefix_cow_pages = cache.cow_pages;
        snapshot.prefix_radix_lookups = cache.radix_lookups;
        snapshot.prefix_radix_nodes = prefix_cache_->radix_nodes();
        snapshot.prefix_cow_bytes_copied = cache.cow_bytes_copied;
        snapshot.prefix_cow_bytes_saved = cache.cow_bytes_saved;
    }
    snapshot.logical_pages_used = paged_kv_ ? paged_kv_->used_pages() : 0;
    return snapshot;
}

GroupedConcurrentMetrics ConcurrentEngine::Impl::grouped_metrics() const {
    return group_concurrent_metrics(metrics());
}


ConcurrentEngine::ConcurrentEngine(std::string safetensors_path,
                                   int max_context,
                                   ModelOptions model_options,
                                   ConcurrentEngineOptions engine_options)
    : impl_(std::make_unique<Impl>(std::move(safetensors_path), max_context,
                                   model_options, engine_options)) {}

ConcurrentEngine::~ConcurrentEngine() = default;

ConcurrentEngine::RequestId ConcurrentEngine::submit(
    std::vector<int32_t> prompt, ConcurrentRequestOptions options) {
    return impl_->submit(std::move(prompt), std::move(options));
}

bool ConcurrentEngine::cancel(RequestId id) { return impl_->cancel(id); }

PollResult ConcurrentEngine::poll(RequestId id, size_t max_tokens) {
    return impl_->poll(id, max_tokens);
}

RequestStatus ConcurrentEngine::status(RequestId id) const {
    return impl_->status(id);
}

bool ConcurrentEngine::step() { return impl_->step(); }
void ConcurrentEngine::start() { impl_->start(); }
void ConcurrentEngine::stop() { impl_->stop(); }
bool ConcurrentEngine::running() const { return impl_->running(); }
ConcurrentMetrics ConcurrentEngine::metrics() const { return impl_->metrics(); }
GroupedConcurrentMetrics ConcurrentEngine::grouped_metrics() const {
    return impl_->grouped_metrics();
}


} // namespace lfm
