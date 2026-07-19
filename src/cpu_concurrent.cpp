#include "lfm/cpu_concurrent.hpp"

#include "lfm/cpu_packed.hpp"
#include "lfm/cpu_prefix_cache.hpp"
#include "lfm/cpu_numa.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace lfm {
namespace {
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

bool terminal(CpuRequestStatus status) {
    return status == CpuRequestStatus::Completed ||
           status == CpuRequestStatus::Cancelled ||
           status == CpuRequestStatus::Failed;
}
} // namespace

double CpuConcurrentMetrics::prefill_tokens_per_second() const {
    return cumulative_prefill_ms > 0.0
        ? static_cast<double>(prefill_tokens) * 1000.0 / cumulative_prefill_ms
        : 0.0;
}

double CpuConcurrentMetrics::decode_tokens_per_second() const {
    return cumulative_decode_ms > 0.0
        ? static_cast<double>(decode_tokens) * 1000.0 / cumulative_decode_ms
        : 0.0;
}

double CpuConcurrentMetrics::average_ttft_ms() const {
    return ttft_samples ? cumulative_ttft_ms / static_cast<double>(ttft_samples) : 0.0;
}

double CpuConcurrentMetrics::average_itl_ms() const {
    return itl_samples ? cumulative_itl_ms / static_cast<double>(itl_samples) : 0.0;
}

const char* cpu_request_status_name(CpuRequestStatus status) {
    switch (status) {
        case CpuRequestStatus::Queued: return "queued";
        case CpuRequestStatus::Prefilling: return "prefilling";
        case CpuRequestStatus::Decoding: return "decoding";
        case CpuRequestStatus::Completed: return "completed";
        case CpuRequestStatus::Cancelled: return "cancelled";
        case CpuRequestStatus::Failed: return "failed";
    }
    return "unknown";
}

struct CpuConcurrentEngine::Impl {
    struct Request {
        CpuRequestId id = 0;
        CpuRequestStatus status = CpuRequestStatus::Queued;
        std::vector<int32_t> prompt;
        size_t prompt_offset = 0;
        CpuRequestOptions options;
        std::unique_ptr<CpuModel> session;
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

    Impl(const std::string& path, int context, CpuModelOptions model_options,
         CpuConcurrentEngineOptions requested)
        : max_context(context), engine_options(std::move(requested)),
          numa_mode(model_options.numa_mode),
          numa_topology(detect_cpu_numa_topology()),
          base_model(path, context, std::move(model_options)) {
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
        worker = std::thread([this] { worker_loop(); });
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
            running = false;
        }
        condition.notify_all();
        if (worker.joinable()) worker.join();
    }

    size_t active_count_locked() const {
        size_t count = 0;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == CpuRequestStatus::Prefilling ||
                request->status == CpuRequestStatus::Decoding) {
                ++count;
            }
        }
        return count;
    }

    size_t queued_count_locked() const {
        size_t count = 0;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == CpuRequestStatus::Queued) ++count;
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
        if (!prefix_cache || request.prefix_inserted || !request.session ||
            request.prompt_offset != request.prompt.size()) return;
        try {
            request.prefix_inserted = prefix_cache->insert(
                request.prompt, request.session->export_prefix_snapshot());
            sync_prefix_metrics_locked();
        } catch (const std::exception& error) {
            last_error_text = std::string("CPU prefix cache insert: ") + error.what();
        }
    }

    void record_attention_parallel_locked(Request& request) {
        if (!request.session) return;
        const uint64_t current = request.session->attention_parallel_calls();
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
            if (request->status == CpuRequestStatus::Queued) queued.push_back(request);
        }
        std::stable_sort(queued.begin(), queued.end(), [](const auto& a, const auto& b) {
            if (a->options.priority != b->options.priority) {
                return a->options.priority > b->options.priority;
            }
            return a->sequence < b->sequence;
        });
        for (const auto& request : queued) {
            if (available == 0) break;
            try {
                request->numa_node = choose_numa_node_locked();
                request->session = base_model.clone_session_on_node(request->numa_node);
                request->session->set_generation_config(request->options.generation);
                request->status = CpuRequestStatus::Prefilling;
                if (prefix_cache) {
                    if (auto match = prefix_cache->acquire(
                            request->prompt, request->numa_node)) {
                        const bool exact = match->matched_tokens == request->prompt.size();
                        request->session->restore_prefix_snapshot(
                            std::move(match->snapshot), exact);
                        request->prompt_offset = match->matched_tokens;
                        request->status = exact ? CpuRequestStatus::Decoding
                                                : CpuRequestStatus::Prefilling;
                    }
                    sync_prefix_metrics_locked();
                }
                --available;
            } catch (const std::exception& error) {
                request->status = CpuRequestStatus::Failed;
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
            if (request->status == CpuRequestStatus::Decoding) result.push_back(request);
        }
        std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            if (a->options.priority != b->options.priority) {
                return a->options.priority > b->options.priority;
            }
            return a->sequence < b->sequence;
        });
        if (result.size() > limit) result.resize(limit);
        return result;
    }

    std::vector<std::shared_ptr<Request>> plan_prefill_locked(size_t limit) {
        std::vector<std::shared_ptr<Request>> result;
        for (const auto& [id, request] : requests) {
            (void)id;
            if (request->status == CpuRequestStatus::Prefilling &&
                request->prompt_offset < request->prompt.size()) {
                result.push_back(request);
            }
        }
        std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            if (a->options.priority != b->options.priority) {
                return a->options.priority > b->options.priority;
            }
            return a->sequence < b->sequence;
        });
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
            auto [tokens, step_metrics] = executor.decode_step(sessions);
            const Clock::time_point now = Clock::now();
            std::lock_guard lock(mutex);
            metrics.packed_decode_steps++;
            metrics.decode_tokens += tokens.size();
            metrics.maximum_decode_batch = std::max<uint64_t>(
                metrics.maximum_decode_batch, tokens.size());
            metrics.cumulative_decode_ms += step_metrics.elapsed_ms;
            for (size_t index = 0; index < plan.size(); ++index) {
                Request& request = *plan[index];
                if (request.status == CpuRequestStatus::Cancelled) continue;
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
                if (token == request.options.eos_token_id ||
                    request.generated.size() >= request.options.max_new_tokens) {
                    request.status = CpuRequestStatus::Completed;
                    request.session.reset();
                    ++metrics.completed_requests;
                }
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                if (!terminal(request->status)) {
                    request->status = CpuRequestStatus::Failed;
                    request->error = error.what();
                    request->session.reset();
                    ++metrics.failed_requests;
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
        size_t chunk_tokens = 0;
        {
            std::lock_guard lock(mutex);
            auto candidates = plan_prefill_locked(engine_options.max_active_requests);
            if (candidates.size() == 1) {
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
            if (!chunked) {
                const size_t limit = std::min({budget,
                    engine_options.max_prefill_batch,
                    engine_options.max_active_requests});
                if (candidates.size() > limit) candidates.resize(limit);
                plan = std::move(candidates);
            }
        }
        if (plan.empty()) return false;

        if (chunked) {
            const auto& request = plan.front();
            const size_t offset = request->prompt_offset;
            const bool final_chunk = offset + chunk_tokens == request->prompt.size();
            try {
                const CpuPackedStepMetrics step_metrics = executor.prefill_chunk(
                    request->session.get(),
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
                if (request->status != CpuRequestStatus::Cancelled) {
                    request->prompt_offset += chunk_tokens;
                    if (request->prompt_offset == request->prompt.size()) {
                        request->status = CpuRequestStatus::Decoding;
                        cache_completed_prefix_locked(*request);
                    }
                }
                refresh_counts_locked();
            } catch (const std::exception& error) {
                std::lock_guard lock(mutex);
                last_error_text = error.what();
                if (!terminal(request->status)) {
                    request->status = CpuRequestStatus::Failed;
                    request->error = error.what();
                    request->session.reset();
                    ++metrics.failed_requests;
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
            const CpuPackedStepMetrics step_metrics = executor.prefill_step(items);
            std::lock_guard lock(mutex);
            ++metrics.ragged_prefill_steps;
            metrics.prefill_tokens += items.size();
            metrics.maximum_prefill_batch = std::max<uint64_t>(
                metrics.maximum_prefill_batch, items.size());
            metrics.cumulative_prefill_ms += step_metrics.elapsed_ms;
            for (const auto& request : plan) {
                if (request->status == CpuRequestStatus::Cancelled) continue;
                record_attention_parallel_locked(*request);
                ++request->prompt_offset;
                if (request->prompt_offset == request->prompt.size()) {
                    request->status = CpuRequestStatus::Decoding;
                    cache_completed_prefix_locked(*request);
                }
            }
            refresh_counts_locked();
        } catch (const std::exception& error) {
            std::lock_guard lock(mutex);
            last_error_text = error.what();
            for (const auto& request : plan) {
                if (!terminal(request->status)) {
                    request->status = CpuRequestStatus::Failed;
                    request->error = error.what();
                    request->session.reset();
                    ++metrics.failed_requests;
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

    void worker_loop() {
        std::unique_lock lock(mutex);
        while (!stopping) {
            condition.wait(lock, [&] {
                return stopping || (running && (queued_count_locked() > 0 ||
                    active_count_locked() > 0));
            });
            if (stopping) break;
            lock.unlock();
            const bool progressed = step_once();
            lock.lock();
            if (!progressed && !stopping) {
                condition.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }

    int max_context = 0;
    CpuConcurrentEngineOptions engine_options;
    CpuNumaMode numa_mode = CpuNumaMode::Disabled;
    CpuNumaTopology numa_topology;
    size_t next_numa_node = 0;
    CpuModel base_model;
    CpuPackedExecutor executor;
    std::unique_ptr<CpuPrefixCacheManager> prefix_cache;

    mutable std::mutex mutex;
    std::mutex execution_mutex;
    std::condition_variable condition;
    std::unordered_map<CpuRequestId, std::shared_ptr<Request>> requests;
    CpuRequestId next_id = 1;
    uint64_t next_sequence = 1;
    CpuConcurrentMetrics metrics;
    std::string last_error_text;
    bool running = false;
    bool stopping = false;
    std::thread worker;
};

CpuConcurrentEngine::CpuConcurrentEngine(
    const std::string& path, int max_context, CpuModelOptions model_options,
    CpuConcurrentEngineOptions engine_options)
    : impl_(std::make_unique<Impl>(path, max_context, std::move(model_options),
                                   std::move(engine_options))) {}

CpuConcurrentEngine::~CpuConcurrentEngine() = default;

CpuRequestId CpuConcurrentEngine::submit(std::vector<int32_t> prompt,
                                         CpuRequestOptions options) {
    options.generation.validate();
    if (prompt.empty()) throw std::invalid_argument("CPU request prompt is empty");
    if (options.max_new_tokens == 0) {
        throw std::invalid_argument("CPU request max_new_tokens must be positive");
    }
    if (prompt.size() + options.max_new_tokens >
        static_cast<size_t>(impl_->max_context)) {
        throw std::invalid_argument("CPU request exceeds configured context");
    }
    for (int32_t token : prompt) {
        if (token < 0 || token >= impl_->base_model.vocab_size()) {
            throw std::invalid_argument("CPU request token out of range");
        }
    }
    auto request = std::make_shared<Impl::Request>();
    {
        std::lock_guard lock(impl_->mutex);
        request->id = impl_->next_id++;
        request->sequence = impl_->next_sequence++;
        request->prompt = std::move(prompt);
        request->options = options;
        request->submitted_at = Clock::now();
        impl_->requests.emplace(request->id, request);
        ++impl_->metrics.submitted_requests;
        impl_->refresh_counts_locked();
    }
    impl_->condition.notify_all();
    return request->id;
}

std::vector<int32_t> CpuConcurrentEngine::poll(CpuRequestId id,
                                               size_t max_tokens,
                                               bool* finished) {
    if (max_tokens == 0) throw std::invalid_argument("poll max_tokens must be positive");
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->requests.find(id);
    if (it == impl_->requests.end()) throw std::out_of_range("unknown CPU request id");
    Impl::Request& request = *it->second;
    const size_t available = request.generated.size() - request.poll_offset;
    const size_t count = std::min(max_tokens, available);
    std::vector<int32_t> result(
        request.generated.begin() + static_cast<ptrdiff_t>(request.poll_offset),
        request.generated.begin() + static_cast<ptrdiff_t>(request.poll_offset + count));
    request.poll_offset += count;
    if (finished) {
        *finished = terminal(request.status) &&
                    request.poll_offset == request.generated.size();
    }
    return result;
}

CpuRequestStatus CpuConcurrentEngine::status(CpuRequestId id) const {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->requests.find(id);
    if (it == impl_->requests.end()) throw std::out_of_range("unknown CPU request id");
    return it->second->status;
}

void CpuConcurrentEngine::cancel(CpuRequestId id) {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->requests.find(id);
    if (it == impl_->requests.end()) throw std::out_of_range("unknown CPU request id");
    Impl::Request& request = *it->second;
    if (!terminal(request.status)) {
        request.status = CpuRequestStatus::Cancelled;
        request.session.reset();
        ++impl_->metrics.cancelled_requests;
        impl_->refresh_counts_locked();
    }
}

bool CpuConcurrentEngine::step() { return impl_->step_once(); }

void CpuConcurrentEngine::start() {
    {
        std::lock_guard lock(impl_->mutex);
        impl_->running = true;
    }
    impl_->condition.notify_all();
}

void CpuConcurrentEngine::stop() {
    std::lock_guard lock(impl_->mutex);
    impl_->running = false;
}

CpuConcurrentMetrics CpuConcurrentEngine::metrics() const {
    std::lock_guard lock(impl_->mutex);
    CpuConcurrentMetrics result = impl_->metrics;
    if (impl_->prefix_cache) {
        const auto& source = impl_->prefix_cache->metrics();
        result.prefix_cache_hits = source.hits;
        result.prefix_cache_misses = source.misses;
        result.prefix_cache_partial_hits = source.partial_hits;
        result.prefix_cache_inserts = source.inserts;
        result.prefix_cache_evictions = source.evictions;
        result.prefix_reused_tokens = source.reused_tokens;
        result.prefix_cow_pages = source.cow_pages;
        result.prefix_cow_bytes = source.cow_bytes;
    }
    result.active_requests = impl_->active_count_locked();
    result.queued_requests = impl_->queued_count_locked();
    return result;
}

std::string CpuConcurrentEngine::backend_description() const {
    return impl_->base_model.backend_description() + " continuous-batching prefix-radix numa-aware";
}

std::string CpuConcurrentEngine::last_error() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->last_error_text;
}

} // namespace lfm
