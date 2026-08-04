#include "celeg/backend/cpu/concurrent.hpp"

#include "celeg/backend/cpu/prefix_cache.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/detail/runtime/concurrency/worker.hpp"
#include "celeg/detail/runtime/concurrency/batch_planner.hpp"

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

} // namespace

struct CpuSchedulerDriver {
    struct Request {
        RequestId id = 0;
        RequestStatus status = RequestStatus::Queued;
        std::vector<int32_t> prompt;
        size_t prompt_offset = 0;
        ConcurrentRequestOptions options;
        std::unique_ptr<CpuModel> session;
        bool cancel_requested = false;
        std::vector<int32_t> generated;
        size_t poll_offset = 0;
        uint64_t sequence = 0;
        Clock::time_point submitted_at;
        Clock::time_point last_token_at;
        bool first_token_recorded = false;
        bool prefix_inserted = false;
        uint64_t attention_parallel_observed = 0;
        int numa_node = -1;
        std::string error;
    };

    CpuSchedulerDriver(const std::string& path, int context, CpuModelOptions model_options,
         CpuConcurrentEngineOptions requested,
         std::shared_ptr<const RuntimeContext> runtime)
        : max_context(context), engine_options(std::move(requested)),
          numa_mode(model_options.numa_mode),
          numa_topology(detect_cpu_numa_topology()),
          runtime_(runtime ? std::move(runtime) : create_builtin_runtime_context()),
          base_model(path, context, std::move(model_options), {}, runtime_) {
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
        // Start only after validation, so constructor failure cannot destroy a
        // joinable std::thread during stack unwinding.
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

    int choose_numa_node_locked() {
        if (numa_mode == CpuNumaMode::Disabled || numa_topology.node_count() <= 1) {
            return -1;
        }
        const int node = static_cast<int>(next_numa_node % numa_topology.node_count());
        ++next_numa_node;
        return node;
    }

    void sync_prefix_metrics_locked() {
        if (!prefix_cache) return;
        const auto& source = prefix_cache->metrics();
        metrics.prefix_cache_hits = source.hits;
        metrics.prefix_cache_misses = source.misses;
        metrics.prefix_cache_partial_hits = source.partial_hits;
        metrics.prefix_cache_inserts = source.inserts;
        metrics.prefix_cache_evictions = source.evictions;
        metrics.prefix_reused_tokens = source.reused_tokens;
        metrics.prefix_cow_pages = source.cow_pages;
        metrics.prefix_cow_bytes = source.cow_bytes;
    }

    void cache_completed_prefix_locked(Request& request) {
        if (!prefix_cache || !request.options.prompt_embedding.empty() ||
            request.prefix_inserted || !request.session ||
            request.prompt_offset != request.prompt.size()) return;
        try {
            request.prefix_inserted = prefix_cache->insert(
                request.prompt, request.session->persistence().export_prefix_snapshot());
            sync_prefix_metrics_locked();
        } catch (const std::exception& error) {
            last_error_text = std::string("CPU prefix cache insert: ") + error.what();
        }
    }

    void record_attention_parallel_locked(Request& request) {
        if (!request.session) return;
        const uint64_t current = request.session->diagnostics().attention_parallel_calls();
        if (current >= request.attention_parallel_observed) {
            metrics.attention_parallel_calls +=
                current - request.attention_parallel_observed;
        }
        request.attention_parallel_observed = current;
    }

    void admit_locked() {
        size_t available = engine_options.max_active_requests -
                           std::min(engine_options.max_active_requests,
                                    active_count_locked());
        if (available == 0) return;
        std::vector<std::shared_ptr<Request>> queued;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == RequestStatus::Queued) queued.push_back(request);
        }
        std::vector<detail::RequestPriorityView> queued_views;
        queued_views.reserve(queued.size());
        for (const auto& request : queued) {
            queued_views.push_back({request->id, request->options.priority});
        }
        const auto queued_order = planner.order_priority(queued_views);
        std::vector<std::shared_ptr<Request>> ordered_queued;
        ordered_queued.reserve(queued.size());
        for (const size_t index : queued_order) ordered_queued.push_back(queued[index]);
        queued = std::move(ordered_queued);
        for (const auto& request : queued) {
            if (available == 0) break;
            try {
                request->numa_node = choose_numa_node_locked();
                request->session = base_model.clone_session_on_node(request->numa_node);
                request->session->session().set_generation_config(request->options.generation);
                request->status = RequestStatus::Prefill;
                if (prefix_cache && request->options.prompt_embedding.empty()) {
                    if (auto match = prefix_cache->acquire(
                            request->prompt, request->numa_node)) {
                        const bool exact = match->matched_tokens == request->prompt.size();
                        request->session->persistence().restore_prefix_snapshot(
                            std::move(match->snapshot), exact);
                        request->prompt_offset = match->matched_tokens;
                        request->status = exact ? RequestStatus::Decoding
                                                : RequestStatus::Prefill;
                    }
                    sync_prefix_metrics_locked();
                }
                --available;
            } catch (const std::exception& error) {
                request->status = RequestStatus::Failed;
                request->error = error.what();
                ++metrics.failed_requests;
            }
        }
        refresh_counts_locked();
    }

    std::vector<std::shared_ptr<Request>> plan_decode_locked(size_t limit) {
        std::vector<std::shared_ptr<Request>> result;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == RequestStatus::Decoding) result.push_back(request);
        }
        std::vector<detail::RequestPriorityView> views;
        views.reserve(result.size());
        for (const auto& request : result) {
            views.push_back({request->id, request->options.priority});
        }
        const auto order = planner.order_priority(views);
        std::vector<std::shared_ptr<Request>> ordered;
        ordered.reserve(result.size());
        for (const size_t index : order) ordered.push_back(result[index]);
        result = std::move(ordered);
        if (result.size() > limit) result.resize(limit);
        return result;
    }

    std::vector<std::shared_ptr<Request>> plan_prefill_locked(size_t limit) {
        std::vector<std::shared_ptr<Request>> result;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == RequestStatus::Prefill &&
                request->prompt_offset < request->prompt.size()) {
                result.push_back(request);
            }
        }
        std::vector<detail::RequestPriorityView> views;
        views.reserve(result.size());
        for (const auto& request : result) {
            views.push_back({request->id, request->options.priority});
        }
        const auto order = planner.order_priority(views);
        std::vector<std::shared_ptr<Request>> ordered;
        ordered.reserve(result.size());
        for (const size_t index : order) ordered.push_back(result[index]);
        result = std::move(ordered);
        if (result.size() > limit) result.resize(limit);
        return result;
    }

    bool run_decode_batch(size_t& budget) {
        std::vector<std::shared_ptr<Request>> plan;
        {
            std::lock_guard lock(mutex);
            plan = plan_decode_locked(std::min({budget,
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
            metrics.packed_decode_steps++;
            metrics.decode_tokens += tokens.size();
            metrics.maximum_decode_batch = std::max<uint64_t>(
                metrics.maximum_decode_batch, tokens.size());
            metrics.cumulative_decode_ms += step_metrics.elapsed_ms;
            for (size_t index = 0; index < plan.size(); ++index) {
                Request& request = *plan[index];
                if (request.status == RequestStatus::Cancelled) continue;
                if (request.cancel_requested) {
                    request.status = RequestStatus::Cancelled;
                    request.session.reset();
                    ++metrics.cancelled_requests;
                    continue;
                }
                const int32_t token = tokens[index];
                request.generated.push_back(token);
                if (!request.first_token_recorded) {
                    request.first_token_recorded = true;
                    metrics.cumulative_ttft_ms += milliseconds(now - request.submitted_at);
                    ++metrics.ttft_samples;
                } else {
                    metrics.cumulative_itl_ms += milliseconds(now - request.last_token_at);
                    ++metrics.itl_samples;
                }
                request.last_token_at = now;
                record_attention_parallel_locked(request);
                if (is_stop_token(request.options.eos_tokens, token) ||
                    request.generated.size() >= request.options.max_new_tokens) {
                    request.status = RequestStatus::Finished;
                    request.session.reset();
                    ++metrics.completed_requests;
                }
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                if (!is_terminal(request->status)) {
                    request->status = request->cancel_requested
                        ? RequestStatus::Cancelled : RequestStatus::Failed;
                    if (request->status == RequestStatus::Failed) {
                        request->error = error.what();
                    }
                    request->session.reset();
                    if (request->status == RequestStatus::Failed) {
                        ++metrics.failed_requests;
                    } else {
                        ++metrics.cancelled_requests;
                    }
                }
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
            auto candidates = plan_prefill_locked(engine_options.max_active_requests);
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
                if (request->cancel_requested) {
                    request->status = RequestStatus::Cancelled;
                    request->session.reset();
                    ++metrics.cancelled_requests;
                } else if (request->status != RequestStatus::Cancelled) {
                    request->prompt_offset = request->prompt.size();
                    request->status = RequestStatus::Decoding;
                    metrics.prefill_tokens += request->prompt.size();
                    cache_completed_prefix_locked(*request);
                }
                refresh_counts_locked();
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex);
                last_error_text = error.what();
                request->status = request->cancel_requested
                    ? RequestStatus::Cancelled : RequestStatus::Failed;
                request->error = error.what();
                request->session.reset();
                if (request->status == RequestStatus::Failed) ++metrics.failed_requests;
                else ++metrics.cancelled_requests;
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
                ++metrics.chunked_prefill_steps;
                metrics.chunked_prefill_tokens += chunk_tokens;
                metrics.prefill_tokens += chunk_tokens;
                metrics.maximum_prefill_chunk = std::max<uint64_t>(
                    metrics.maximum_prefill_chunk, chunk_tokens);
                metrics.cumulative_prefill_ms += step_metrics.elapsed_ms;
                record_attention_parallel_locked(*request);
                if (request->cancel_requested) {
                    request->status = RequestStatus::Cancelled;
                    request->session.reset();
                    ++metrics.cancelled_requests;
                } else if (request->status != RequestStatus::Cancelled) {
                    request->prompt_offset += chunk_tokens;
                    if (request->prompt_offset == request->prompt.size()) {
                        request->status = RequestStatus::Decoding;
                        cache_completed_prefix_locked(*request);
                    }
                }
                refresh_counts_locked();
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex);
                last_error_text = error.what();
                if (!is_terminal(request->status)) {
                    request->status = request->cancel_requested
                        ? RequestStatus::Cancelled : RequestStatus::Failed;
                    if (request->status == RequestStatus::Failed) {
                        request->error = error.what();
                    }
                    request->session.reset();
                    if (request->status == RequestStatus::Failed) {
                        ++metrics.failed_requests;
                    } else {
                        ++metrics.cancelled_requests;
                    }
                }
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
            ++metrics.ragged_prefill_steps;
            metrics.prefill_tokens += items.size();
            metrics.maximum_prefill_batch = std::max<uint64_t>(
                metrics.maximum_prefill_batch, items.size());
            metrics.cumulative_prefill_ms += step_metrics.elapsed_ms;
            for (const auto& request : plan) {
                if (request->status == RequestStatus::Cancelled) continue;
                if (request->cancel_requested) {
                    request->status = RequestStatus::Cancelled;
                    request->session.reset();
                    ++metrics.cancelled_requests;
                    continue;
                }
                record_attention_parallel_locked(*request);
                ++request->prompt_offset;
                if (request->prompt_offset == request->prompt.size()) {
                    request->status = RequestStatus::Decoding;
                    cache_completed_prefix_locked(*request);
                }
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                if (!is_terminal(request->status)) {
                    request->status = request->cancel_requested
                        ? RequestStatus::Cancelled : RequestStatus::Failed;
                    if (request->status == RequestStatus::Failed) {
                        request->error = error.what();
                    }
                    request->session.reset();
                    if (request->status == RequestStatus::Failed) {
                        ++metrics.failed_requests;
                    } else {
                        ++metrics.cancelled_requests;
                    }
                }
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
            admit_locked();
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
            // Decode requests that became ready during the ragged prefill waves.
            progressed |= run_decode_batch(budget);
        }
        {
            std::lock_guard lock(mutex);
            admit_locked();
            metrics.cumulative_scheduler_ms += milliseconds(Clock::now() - started);
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
    CpuNumaMode numa_mode = CpuNumaMode::Disabled;
    CpuNumaTopology numa_topology;
    size_t next_numa_node = 0;
    std::shared_ptr<const RuntimeContext> runtime_;
    CpuModel base_model;
    std::unique_ptr<CpuPrefixCacheManager> prefix_cache;
    detail::BatchPlanner planner;

    mutable std::mutex mutex;
    std::mutex execution_mutex;
    std::unordered_map<RequestId, std::shared_ptr<Request>> requests;
    RequestId next_id = 1;
    uint64_t next_sequence = 1;
    CpuConcurrentMetrics metrics;
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
        ++state_->metrics.submitted_requests;
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
        ++state_->metrics.cancelled_requests;
        state_->refresh_counts_locked();
    } else {
        // Active execution owns the session until the scheduler returns.
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

CpuConcurrentMetrics CpuConcurrentEngine::metrics() const {
    std::lock_guard lock(state_->mutex);
    CpuConcurrentMetrics result = state_->metrics;
    if (state_->prefix_cache) {
        const auto& source = state_->prefix_cache->metrics();
        result.prefix_cache_hits = source.hits;
        result.prefix_cache_misses = source.misses;
        result.prefix_cache_partial_hits = source.partial_hits;
        result.prefix_cache_inserts = source.inserts;
        result.prefix_cache_evictions = source.evictions;
        result.prefix_reused_tokens = source.reused_tokens;
        result.prefix_cow_pages = source.cow_pages;
        result.prefix_cow_bytes = source.cow_bytes;
    }
    result.active_requests = state_->active_count_locked();
    result.queued_requests = state_->queued_count_locked();
    return result;
}

std::string CpuConcurrentEngine::backend_description() const {
    return state_->base_model.diagnostics().backend_description() + " continuous-batching prefix-radix numa-aware";
}

std::string CpuConcurrentEngine::last_error() const {
    std::lock_guard lock(state_->mutex);
    return state_->last_error_text;
}

} // namespace celeg
