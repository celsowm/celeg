#include "celeg/backend/cpu/concurrent.hpp"

#include "celeg/backend/cpu/prefix_cache.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/backend/cpu/numa_placement.hpp"
#include "celeg/detail/runtime/concurrency/worker.hpp"
#include "celeg/detail/runtime/concurrency/batch_planner.hpp"
#include "detail/concurrent_request.hpp"
#include "detail/batch_scheduler.hpp"
#include "detail/admission_controller.hpp"
#include "detail/request_lifecycle.hpp"
#include "detail/metrics_collector.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace celeg {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

}

struct CpuSchedulerDriver {
    using Request = CpuConcurrentRequest;

    CpuSchedulerDriver(const std::string& path, int context, CpuModelOptions model_options,
         CpuConcurrentEngineOptions requested,
         std::shared_ptr<const RuntimeContext> runtime)
        : max_context(context), engine_options(std::move(requested)),
          numa_placement(model_options.numa_mode, detect_cpu_numa_topology()),
          runtime_(runtime ? std::move(runtime) : create_builtin_runtime_context()),
          base_model(path, context, std::move(model_options), {}, runtime_),
          admission(base_model, numa_placement, prefix_cache),
          metrics_collector(metrics, metrics_extras) {
        if (engine_options.max_active_requests == 0 ||
            engine_options.max_batched_tokens == 0 ||
            engine_options.max_prefill_batch == 0 ||
            engine_options.max_decode_batch == 0 ||
            engine_options.long_prefill_chunk_tokens == 0 ||
            engine_options.long_prefill_threshold == 0 ||
            (engine_options.prefix_cache &&
             (engine_options.prefix_cache_max_entries == 0 ||
              engine_options.prefix_cache_max_bytes == 0))) {
            throw std::invalid_argument("CPU concurrent engine limits must be positive");
        }
        if (engine_options.prefix_cache) {
            prefix_cache = std::make_unique<CpuPrefixCacheManager>(
                base_model.shared_kv_pools(),
                engine_options.prefix_cache_max_entries,
                engine_options.prefix_cache_max_bytes);
        }
        if (engine_options.worker_thread) {
            engine_worker.start([this] { return worker_step(); }, 1000);
        }
    }

    ~CpuSchedulerDriver() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            running = false;
        }
        engine_worker.stop();
    }

    size_t active_count_locked() const {
        size_t count = 0;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == RequestStatus::Prefill ||
                request->status == RequestStatus::Decoding) {
                ++count;
            }
        }
        return count;
    }

    size_t queued_count_locked() const {
        size_t count = 0;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == RequestStatus::Queued) ++count;
        }
        return count;
    }

    void refresh_counts_locked() {
        metrics.active_requests = active_count_locked();
        metrics.queued_requests = queued_count_locked();
    }

    void cache_completed_prefix_locked(Request& request) {
        if (!prefix_cache || !request.options.prompt_embedding.empty() ||
            request.prefix_inserted || !request.session ||
            request.prompt_offset != request.prompt.size()) return;
        try {
            request.prefix_inserted = prefix_cache->insert(
                request.prompt, request.session->persistence().export_prefix_snapshot());
            admission.sync_prefix_metrics(metrics);
        } catch (const std::exception& error) {
            last_error_text = std::string("CPU prefix cache insert: ") + error.what();
        }
    }

    bool run_decode_batch(size_t& budget) {
        std::vector<std::shared_ptr<Request>> plan;
        {
            std::lock_guard lock(mutex);
            plan = CpuBatchScheduler::plan_decode(requests, planner, std::min({budget,
                engine_options.max_decode_batch,
                engine_options.max_active_requests}));
        }
        if (plan.empty()) return false;
        std::vector<CpuModel*> sessions;
        sessions.reserve(plan.size());
        for (const auto& request : plan) sessions.push_back(request->session.get());
        try {
            auto [tokens, step_metrics] = CpuModel::decode_batch(sessions);
            const Clock::time_point now = Clock::now();
            std::lock_guard lock(mutex);
            metrics_collector.record_decode_batch(tokens.size(), step_metrics);
            for (size_t index = 0; index < plan.size(); ++index) {
                CpuRequestLifecycle::apply_decode_token(
                    plan[index], tokens[index], now, metrics, metrics_extras);
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                CpuRequestLifecycle::fail(request, error.what(), metrics);
            }
            refresh_counts_locked();
        }
        budget -= plan.size();
        return true;
    }

    bool run_prefill_batch(size_t& budget) {
        std::vector<std::shared_ptr<Request>> plan;
        bool chunked = false;
        bool embedded = false;
        size_t chunk_tokens = 0;
        {
            std::lock_guard lock(mutex);
            auto candidates = CpuBatchScheduler::plan_prefill(
                requests, planner, engine_options.max_active_requests);
            for (const auto& candidate : candidates) {
                if (!candidate->options.prompt_embedding.empty()) {
                    plan = {candidate};
                    embedded = true;
                    break;
                }
            }
            if (embedded) {
                budget = 0;
            }
            if (!embedded && candidates.size() == 1) {
                const auto& request = candidates.front();
                const size_t remaining = request->prompt.size() - request->prompt_offset;
                if (remaining >= engine_options.long_prefill_threshold &&
                    budget >= engine_options.long_prefill_threshold) {
                    chunked = true;
                    chunk_tokens = std::min({remaining, budget,
                        engine_options.long_prefill_chunk_tokens});
                    plan = std::move(candidates);
                }
            }
            if (!embedded && !chunked) {
                const size_t limit = std::min({budget,
                    engine_options.max_prefill_batch,
                    engine_options.max_active_requests});
                if (candidates.size() > limit) candidates.resize(limit);
                plan = std::move(candidates);
            }
        }
        if (plan.empty()) return false;

        if (embedded) {
            const auto& request = plan.front();
            try {
                request->session->session().prefill(
                    request->prompt, request->options.prompt_embedding);
                std::lock_guard lock(mutex);
                const size_t consumed = request->prompt.size() - request->prompt_offset;
                CpuRequestLifecycle::apply_prefill(request, consumed, metrics);
                if (request->status == RequestStatus::Decoding) {
                    metrics_collector.record_prefill_tokens(request->prompt.size());
                    cache_completed_prefix_locked(*request);
                }
                refresh_counts_locked();
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex);
                last_error_text = error.what();
                CpuRequestLifecycle::fail(request, error.what(), metrics);
                refresh_counts_locked();
            }
            return true;
        }

        if (chunked) {
            const auto& request = plan.front();
            const size_t offset = request->prompt_offset;
            const bool final_chunk = offset + chunk_tokens == request->prompt.size();
            try {
                const CpuBatchMetrics step_metrics = CpuModel::prefill_chunk(
                    *request->session,
                    std::span<const int32_t>(request->prompt.data() + offset,
                                             chunk_tokens),
                    final_chunk);
                std::lock_guard lock(mutex);
                metrics_collector.record_chunked_prefill(chunk_tokens, step_metrics);
                CpuRequestLifecycle::observe_attention(request, metrics_extras);
                CpuRequestLifecycle::apply_prefill(request, chunk_tokens, metrics);
                if (request->status == RequestStatus::Decoding) {
                    cache_completed_prefix_locked(*request);
                }
                refresh_counts_locked();
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex);
                last_error_text = error.what();
                CpuRequestLifecycle::fail(request, error.what(), metrics);
                refresh_counts_locked();
            }
            budget -= chunk_tokens;
            return true;
        }

        std::vector<CpuPrefillItem> items;
        items.reserve(plan.size());
        for (const auto& request : plan) {
            const size_t offset = request->prompt_offset;
            items.push_back({request->session.get(), request->prompt[offset],
                             offset + 1 == request->prompt.size()});
        }
        try {
            const CpuBatchMetrics step_metrics = CpuModel::prefill_batch(items);
            std::lock_guard lock(mutex);
            metrics_collector.record_ragged_prefill(items.size(), step_metrics);
            for (const auto& request : plan) {
                CpuRequestLifecycle::observe_attention(request, metrics_extras);
                const RequestStatus before = request->status;
                CpuRequestLifecycle::apply_prefill(request, 1, metrics);
                if (before != RequestStatus::Cancelled &&
                    request->status == RequestStatus::Decoding) {
                    cache_completed_prefix_locked(*request);
                }
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                CpuRequestLifecycle::fail(request, error.what(), metrics);
            }
            refresh_counts_locked();
        }
        budget -= plan.size();
        return true;
    }

    bool step_once() {
        std::lock_guard execution_lock(execution_mutex);
        const auto started = Clock::now();
        {
            std::lock_guard lock(mutex);
            admission.admit(requests, planner,
                            engine_options.max_active_requests, metrics);
            refresh_counts_locked();
        }
        size_t budget = engine_options.max_batched_tokens;
        bool progressed = false;
        if (engine_options.decode_first && budget > 0) {
            progressed |= run_decode_batch(budget);
        }
        while (budget > 0) {
            const bool did_prefill = run_prefill_batch(budget);
            progressed |= did_prefill;
            if (!did_prefill) break;
        }
        if (!engine_options.decode_first && budget > 0) {
            progressed |= run_decode_batch(budget);
        } else if (engine_options.decode_first && budget > 0) {
            progressed |= run_decode_batch(budget);
        }
        {
            std::lock_guard lock(mutex);
            admission.admit(requests, planner,
                            engine_options.max_active_requests, metrics);
            metrics_collector.record_scheduler(
                milliseconds(Clock::now() - started));
            refresh_counts_locked();
        }
        return progressed;
    }

    bool worker_step() {
        {
            std::lock_guard lock(mutex);
            if (!running || stopping) return false;
        }
        return step_once();
    }

    int max_context = 0;
    CpuConcurrentEngineOptions engine_options;
    CpuNumaPlacement numa_placement;
    std::shared_ptr<const RuntimeContext> runtime_;
    CpuModel base_model;
    std::unique_ptr<CpuPrefixCacheManager> prefix_cache;
    CpuAdmissionController admission;
    detail::BatchPlanner planner;

    mutable std::mutex mutex;
    std::mutex execution_mutex;
    std::unordered_map<RequestId, std::shared_ptr<Request>> requests;
    RequestId next_id = 1;
    uint64_t next_sequence = 1;
    ConcurrentMetrics metrics;
    CpuConcurrentMetricsExtras metrics_extras;
    CpuMetricsCollector metrics_collector;
    std::string last_error_text;
    bool running = false;
    bool stopping = false;
    detail::EngineWorker engine_worker;
};

CpuConcurrentEngine::CpuConcurrentEngine(
    const std::string& path, int max_context, CpuModelOptions model_options,
    CpuConcurrentEngineOptions engine_options,
    std::shared_ptr<const RuntimeContext> runtime)
    : state_(std::make_unique<CpuSchedulerDriver>(path, max_context, std::move(model_options),
                                   std::move(engine_options), std::move(runtime))) {}

CpuConcurrentEngine::~CpuConcurrentEngine() = default;

RequestId CpuConcurrentEngine::submit(std::vector<int32_t> prompt,
                                      ConcurrentRequestOptions options) {
    options.generation.validate();
    if (prompt.empty()) throw std::invalid_argument("CPU request prompt is empty");
    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("CPU request max_new_tokens must be positive");
    }
    if (prompt.size() + options.max_new_tokens >
        static_cast<size_t>(state_->max_context)) {
        throw std::invalid_argument("CPU request exceeds configured context");
    }
    for (int32_t token : prompt) {
        if (token < 0 || token >= state_->base_model.diagnostics().vocab_size()) {
            throw std::invalid_argument("CPU request token out of range");
        }
    }
    auto request = std::make_shared<CpuSchedulerDriver::Request>();
    {
        std::lock_guard lock(state_->mutex);
        request->id = state_->next_id++;
        request->sequence = state_->next_sequence++;
        request->prompt = std::move(prompt);
        request->options = options;
        request->submitted_at = Clock::now();
        state_->requests.emplace(request->id, request);
        ++state_->metrics.submitted;
        state_->refresh_counts_locked();
    }
    state_->engine_worker.notify();
    return request->id;
}

PollResult CpuConcurrentEngine::poll(RequestId id, size_t max_tokens) {
    std::lock_guard lock(state_->mutex);
    auto it = state_->requests.find(id);
    if (it == state_->requests.end()) throw std::out_of_range("unknown CPU request id");
    CpuSchedulerDriver::Request& request = *it->second;
    const size_t available = request.generated.size() - request.poll_offset;
    const size_t count = max_tokens == 0 ? available : std::min(max_tokens, available);
    std::vector<int32_t> result(
        request.generated.begin() + static_cast<ptrdiff_t>(request.poll_offset),
        request.generated.begin() + static_cast<ptrdiff_t>(request.poll_offset + count));
    request.poll_offset += count;
    PollResult result_value;
    result_value.status = request.status;
    result_value.tokens = std::move(result);
    result_value.finished = is_terminal(request.status) &&
                            request.poll_offset == request.generated.size();
    result_value.error = request.error;
    return result_value;
}

RequestStatus CpuConcurrentEngine::status(RequestId id) const {
    std::lock_guard lock(state_->mutex);
    auto it = state_->requests.find(id);
    if (it == state_->requests.end()) throw std::out_of_range("unknown CPU request id");
    return it->second->status;
}

bool CpuConcurrentEngine::cancel(RequestId id) {
    std::lock_guard lock(state_->mutex);
    auto it = state_->requests.find(id);
    if (it == state_->requests.end()) return false;
    CpuSchedulerDriver::Request& request = *it->second;
    if (is_terminal(request.status)) return false;
    if (request.status == RequestStatus::Queued) {
        request.status = RequestStatus::Cancelled;
        ++state_->metrics.cancelled;
        state_->refresh_counts_locked();
    } else {
        request.cancel_requested = true;
    }
    return true;
}

bool CpuConcurrentEngine::release(RequestId id) {
    std::lock_guard lock(state_->mutex);
    auto it = state_->requests.find(id);
    if (it == state_->requests.end()) return false;
    if (!is_terminal(it->second->status)) return false;
    state_->requests.erase(it);
    return true;
}

bool CpuConcurrentEngine::step() { return state_->step_once(); }

void CpuConcurrentEngine::start() {
    {
        std::lock_guard lock(state_->mutex);
        state_->running = true;
    }
    state_->engine_worker.notify();
}

void CpuConcurrentEngine::stop() {
    std::lock_guard lock(state_->mutex);
    state_->running = false;
}

ConcurrentMetrics CpuConcurrentEngine::metrics() const {
    std::lock_guard lock(state_->mutex);
    ConcurrentMetrics result = state_->metrics;
    state_->admission.sync_prefix_metrics(result);
    result.active_requests = state_->active_count_locked();
    result.queued_requests = state_->queued_count_locked();
    return result;
}

CpuConcurrentMetricsExtras CpuConcurrentEngine::metrics_extras() const {
    std::lock_guard lock(state_->mutex);
    return state_->metrics_extras;
}

std::string CpuConcurrentEngine::backend_description() const {
    return state_->base_model.diagnostics().backend_description() + " continuous-batching prefix-radix numa-aware";
}

std::string CpuConcurrentEngine::last_error() const {
    std::lock_guard lock(state_->mutex);
    return state_->last_error_text;
}

}
