#include "lfm/c_api.h"
#include "lfm/concurrent.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct lfm25_engine {
    std::unique_ptr<lfm::ConcurrentEngine> engine;
    mutable std::string error;
};

namespace {
thread_local std::string concurrent_global_error;

bool valid_struct_v2(uint32_t actual, size_t expected) { return actual >= expected; }

lfm::ModelOptions convert_model_options_v2(const lfm25_model_options_v1& source) {
    lfm::ModelOptions result;
    result.fused_residuals = (source.flags & LFM25_OPTION_FUSED_RESIDUALS) != 0;
    result.fast_attention = (source.flags & LFM25_OPTION_FAST_ATTENTION) != 0;
    result.fused_projections = (source.flags & LFM25_OPTION_FUSED_PROJECTIONS) != 0;
    result.fused_sampling = (source.flags & LFM25_OPTION_FUSED_SAMPLING) != 0;
    result.cuda_graph = (source.flags & LFM25_OPTION_CUDA_GRAPH) != 0;
    result.legacy_prefill = (source.flags & LFM25_OPTION_LEGACY_PREFILL) != 0;
    result.lt_autotune = (source.flags & LFM25_OPTION_LT_AUTOTUNE) != 0;
    result.weight_mode = source.weight_mode == LFM25_WEIGHT_INT8
        ? lfm::WeightMode::Int8
        : (source.weight_mode == LFM25_WEIGHT_INT4
            ? lfm::WeightMode::Int4 : lfm::WeightMode::Bf16);
    result.kv_cache_mode = source.kv_cache_mode == LFM25_KV_INT8
        ? lfm::KvCacheMode::Int8 : lfm::KvCacheMode::Bf16;
    result.gemm_backend = source.gemm_backend == LFM25_GEMM_CUBLASLT
        ? lfm::GemmBackend::CublasLt : lfm::GemmBackend::Cublas;
    result.attention_mode = source.attention_mode == LFM25_ATTENTION_SEGMENTED
        ? lfm::AttentionMode::Segmented
        : (source.attention_mode == LFM25_ATTENTION_AUTO
            ? lfm::AttentionMode::Auto : lfm::AttentionMode::Single);
    result.attention_chunk_tokens = source.attention_chunk_tokens;
    result.attention_auto_threshold = source.attention_auto_threshold;
    result.lt_workspace_bytes = static_cast<size_t>(source.lt_workspace_mb) *
                                1024ULL * 1024ULL;
    result.lt_heuristics = source.lt_heuristics;
    return result;
}

lfm::GenerationConfig convert_generation_v2(
    const lfm25_generation_options_v1& source) {
    lfm::GenerationConfig result;
    result.temperature = source.temperature;
    result.top_k = source.top_k;
    result.top_p = source.top_p;
    result.repetition_penalty = source.repetition_penalty;
    result.seed = source.seed;
    return result;
}

int32_t convert_status(lfm::RequestStatus status) {
    switch (status) {
        case lfm::RequestStatus::Queued: return LFM25_REQUEST_QUEUED;
        case lfm::RequestStatus::Prefill: return LFM25_REQUEST_PREFILL;
        case lfm::RequestStatus::Decoding: return LFM25_REQUEST_DECODING;
        case lfm::RequestStatus::Finished: return LFM25_REQUEST_FINISHED;
        case lfm::RequestStatus::Cancelled: return LFM25_REQUEST_CANCELLED;
        case lfm::RequestStatus::Failed: return LFM25_REQUEST_FAILED;
    }
    return LFM25_REQUEST_FAILED;
}

lfm25_status engine_fail(lfm25_engine* engine, lfm25_status status,
                         const std::string& message) noexcept {
    if (engine) engine->error = message;
    else concurrent_global_error = message;
    return status;
}

template <typename Function>
lfm25_status protect_engine(lfm25_engine* engine, Function&& function) noexcept {
    if (!engine || !engine->engine) {
        return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT,
                           "engine handle is null");
    }
    try {
        function();
        engine->error.clear();
        return LFM25_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::out_of_range& error) {
        return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return engine_fail(engine, LFM25_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return engine_fail(engine, LFM25_STATUS_RUNTIME_ERROR,
                           "unknown concurrent runtime error");
    }
}
} // namespace

extern "C" {

void lfm25_engine_options_init(lfm25_engine_options_v2* options) {
    if (!options) return;
    std::memset(options, 0, sizeof(*options));
    options->struct_size = sizeof(*options);
    options->max_context = 4096;
    options->max_active_requests = 8;
    options->max_batched_tokens = 512;
    options->prefill_chunk_tokens = 256;
    options->page_tokens = 16;
    options->scheduler_policy = LFM25_SCHEDULER_GUARANTEED_NO_EVICT;
    options->worker_thread = 1;
    options->idle_sleep_microseconds = 100;
    lfm25_model_options_init(&options->model);
    options->model.max_context = options->max_context;
}

void lfm25_request_options_init(lfm25_request_options_v2* options) {
    if (!options) return;
    std::memset(options, 0, sizeof(*options));
    options->struct_size = sizeof(*options);
    options->max_new_tokens = 128;
    options->eos_token = 7;
    lfm25_generation_options_init(&options->generation);
}

lfm25_engine* lfm25_engine_create(
    const char* safetensors_path, const lfm25_engine_options_v2* options) {
    concurrent_global_error.clear();
    if (!safetensors_path || !options ||
        !valid_struct_v2(options->struct_size, sizeof(*options)) ||
        !valid_struct_v2(options->model.struct_size, sizeof(options->model))) {
        concurrent_global_error = "invalid engine path or options";
        return nullptr;
    }
    try {
        lfm::ConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = options->max_active_requests;
        engine_options.max_batched_tokens = options->max_batched_tokens;
        engine_options.prefill_chunk_tokens = options->prefill_chunk_tokens;
        engine_options.page_tokens = options->page_tokens;
        engine_options.logical_kv_pages = options->logical_kv_pages;
        engine_options.scheduler_policy =
            options->scheduler_policy == LFM25_SCHEDULER_MAX_UTILIZATION
                ? lfm::SchedulerPolicy::MaxUtilization
                : lfm::SchedulerPolicy::GuaranteedNoEvict;
        engine_options.worker_thread = options->worker_thread != 0;
        engine_options.idle_sleep_microseconds = options->idle_sleep_microseconds;
        auto handle = std::make_unique<lfm25_engine>();
        handle->engine = std::make_unique<lfm::ConcurrentEngine>(
            safetensors_path, options->max_context,
            convert_model_options_v2(options->model), engine_options);
        return handle.release();
    } catch (const std::exception& error) {
        concurrent_global_error = error.what();
        return nullptr;
    } catch (...) {
        concurrent_global_error = "unknown concurrent engine creation error";
        return nullptr;
    }
}

void lfm25_engine_destroy(lfm25_engine* engine) { delete engine; }

const char* lfm25_engine_last_error(const lfm25_engine* engine) {
    return engine ? engine->error.c_str() : concurrent_global_error.c_str();
}

lfm25_status lfm25_engine_start(lfm25_engine* engine) {
    return protect_engine(engine, [&] { engine->engine->start(); });
}

lfm25_status lfm25_engine_stop(lfm25_engine* engine) {
    return protect_engine(engine, [&] { engine->engine->stop(); });
}

lfm25_status lfm25_engine_step(lfm25_engine* engine, int* did_work) {
    if (!did_work) return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT,
                                      "did_work pointer is null");
    return protect_engine(engine, [&] { *did_work = engine->engine->step() ? 1 : 0; });
}

lfm25_status lfm25_engine_submit(
    lfm25_engine* engine, const int32_t* prompt_tokens, size_t prompt_count,
    const lfm25_request_options_v2* options, lfm25_request_id* request_id) {
    if (!prompt_tokens || prompt_count == 0 || !options || !request_id ||
        !valid_struct_v2(options->struct_size, sizeof(*options)) ||
        !valid_struct_v2(options->generation.struct_size,
                         sizeof(options->generation))) {
        return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid submit arguments");
    }
    return protect_engine(engine, [&] {
        lfm::ConcurrentRequestOptions request_options;
        request_options.max_new_tokens = options->max_new_tokens;
        request_options.eos_token = options->eos_token;
        request_options.priority = options->priority;
        request_options.generation = convert_generation_v2(options->generation);
        *request_id = engine->engine->submit(
            std::vector<int32_t>(prompt_tokens, prompt_tokens + prompt_count),
            request_options);
    });
}

lfm25_status lfm25_engine_cancel(
    lfm25_engine* engine, lfm25_request_id request_id) {
    return protect_engine(engine, [&] {
        if (!engine->engine->cancel(request_id)) {
            throw std::invalid_argument("request cannot be cancelled");
        }
    });
}

lfm25_status lfm25_engine_request_status(
    const lfm25_engine* engine, lfm25_request_id request_id,
    int32_t* request_status) {
    if (!engine || !engine->engine || !request_status) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid request_status arguments");
    }
    try {
        *request_status = convert_status(engine->engine->status(request_id));
        return LFM25_STATUS_OK;
    } catch (const std::exception& error) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT, error.what());
    }
}

lfm25_status lfm25_engine_poll(
    lfm25_engine* engine, lfm25_request_id request_id,
    int32_t* tokens, size_t token_capacity, size_t* token_count,
    int32_t* request_status, int* finished) {
    if (!token_count || !request_status || !finished ||
        (token_capacity != 0 && !tokens)) {
        return engine_fail(engine, LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid poll arguments");
    }
    return protect_engine(engine, [&] {
        if (token_capacity == 0) {
            *token_count = 0;
            const auto status = engine->engine->status(request_id);
            *request_status = convert_status(status);
            *finished = status == lfm::RequestStatus::Finished ||
                        status == lfm::RequestStatus::Cancelled ||
                        status == lfm::RequestStatus::Failed;
            return;
        }
        lfm::PollResult result = engine->engine->poll(request_id, token_capacity);
        std::copy(result.tokens.begin(), result.tokens.end(), tokens);
        *token_count = result.tokens.size();
        *request_status = convert_status(result.status);
        *finished = result.finished ? 1 : 0;
        if (result.status == lfm::RequestStatus::Failed && !result.error.empty()) {
            engine->error = result.error;
        }
    });
}

lfm25_status lfm25_engine_get_metrics(
    const lfm25_engine* engine, lfm25_engine_metrics_v2* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->submitted = source.submitted;
    metrics->completed = source.completed;
    metrics->cancelled = source.cancelled;
    metrics->failed = source.failed;
    metrics->scheduler_steps = source.scheduler_steps;
    metrics->prefill_tokens = source.prefill_tokens;
    metrics->decoded_tokens = source.decoded_tokens;
    metrics->active_requests = source.active_requests;
    metrics->queued_requests = source.queued_requests;
    metrics->logical_pages_used = source.logical_pages_used;
    metrics->logical_pages_total = source.logical_pages_total;
    metrics->ttft_samples = source.ttft_samples;
    metrics->itl_samples = source.itl_samples;
    metrics->cumulative_ttft_ms = source.cumulative_ttft_ms;
    metrics->cumulative_itl_ms = source.cumulative_itl_ms;
    metrics->average_ttft_ms = source.average_ttft_ms();
    metrics->average_itl_ms = source.average_itl_ms();
    metrics->cumulative_step_ms = source.cumulative_step_ms;
    return LFM25_STATUS_OK;
}


lfm25_status lfm25_engine_get_packed_metrics(
    const lfm25_engine* engine, lfm25_packed_metrics_v1* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid packed metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->packed_decode_steps = source.packed_decode_steps;
    metrics->packed_decode_tokens = source.packed_decode_tokens;
    metrics->lane_decode_tokens = source.lane_decode_tokens;
    metrics->packed_fallback_batches = source.packed_fallback_batches;
    metrics->maximum_packed_batch = source.maximum_packed_batch;
    metrics->cumulative_packed_decode_ms = source.cumulative_packed_decode_ms;
    metrics->packed_tokens_per_second = source.cumulative_packed_decode_ms > 0.0
        ? static_cast<double>(source.packed_decode_tokens) * 1000.0 /
              source.cumulative_packed_decode_ms
        : 0.0;
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_engine_get_packed_metrics_v2(
    const lfm25_engine* engine, lfm25_packed_metrics_v2* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid packed v2 metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->packed_decode_steps = source.packed_decode_steps;
    metrics->packed_decode_tokens = source.packed_decode_tokens;
    metrics->lane_decode_tokens = source.lane_decode_tokens;
    metrics->packed_fallback_batches = source.packed_fallback_batches;
    metrics->maximum_packed_batch = source.maximum_packed_batch;
    metrics->cumulative_packed_decode_ms = source.cumulative_packed_decode_ms;
    metrics->packed_tokens_per_second = source.cumulative_packed_decode_ms > 0.0
        ? static_cast<double>(source.packed_decode_tokens) * 1000.0 /
              source.cumulative_packed_decode_ms
        : 0.0;
    metrics->ragged_prefill_steps = source.ragged_prefill_steps;
    metrics->ragged_prefill_tokens = source.ragged_prefill_tokens;
    metrics->lane_prefill_tokens = source.lane_prefill_tokens;
    metrics->maximum_ragged_prefill_batch = source.maximum_ragged_prefill_batch;
    metrics->cumulative_ragged_prefill_ms = source.cumulative_ragged_prefill_ms;
    metrics->ragged_prefill_tokens_per_second =
        source.cumulative_ragged_prefill_ms > 0.0
            ? static_cast<double>(source.ragged_prefill_tokens) * 1000.0 /
                  source.cumulative_ragged_prefill_ms
            : 0.0;
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_engine_get_paged_kv_metrics(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v1* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid paged KV metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->physical_pages_used = source.logical_pages_used;
    metrics->physical_pages_total = source.logical_pages_total;
    metrics->physical_kv_bytes = source.physical_kv_bytes;
    metrics->prefix_cache_hits = source.prefix_cache_hits;
    metrics->prefix_cache_misses = source.prefix_cache_misses;
    metrics->prefix_cache_inserts = source.prefix_cache_inserts;
    metrics->prefix_cache_evictions = source.prefix_cache_evictions;
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_engine_get_paged_kv_metrics_v2(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v2* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid paged KV v2 metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->physical_pages_used = source.logical_pages_used;
    metrics->physical_pages_total = source.logical_pages_total;
    metrics->physical_kv_bytes = source.physical_kv_bytes;
    metrics->prefix_cache_hits = source.prefix_cache_hits;
    metrics->prefix_cache_misses = source.prefix_cache_misses;
    metrics->prefix_cache_inserts = source.prefix_cache_inserts;
    metrics->prefix_cache_evictions = source.prefix_cache_evictions;
    metrics->prefix_cache_partial_hits = source.prefix_cache_partial_hits;
    metrics->prefix_reused_tokens = source.prefix_reused_tokens;
    metrics->prefix_cow_pages = source.prefix_cow_pages;
    metrics->direct_paged_prefill_tokens = source.direct_paged_prefill_tokens;
    metrics->segmented_paged_decode_steps =
        source.segmented_paged_decode_steps;
    metrics->segmented_paged_decode_tokens =
        source.segmented_paged_decode_tokens;
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_engine_get_paged_kv_metrics_v3(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v3* metrics) {
    if (!engine || !engine->engine || !metrics ||
        !valid_struct_v2(metrics->struct_size, sizeof(*metrics))) {
        return engine_fail(const_cast<lfm25_engine*>(engine),
                           LFM25_STATUS_INVALID_ARGUMENT,
                           "invalid paged KV v3 metrics arguments");
    }
    const lfm::ConcurrentMetrics source = engine->engine->metrics();
    metrics->physical_pages_used = source.logical_pages_used;
    metrics->physical_pages_total = source.logical_pages_total;
    metrics->physical_kv_bytes = source.physical_kv_bytes;
    metrics->prefix_cache_hits = source.prefix_cache_hits;
    metrics->prefix_cache_misses = source.prefix_cache_misses;
    metrics->prefix_cache_inserts = source.prefix_cache_inserts;
    metrics->prefix_cache_evictions = source.prefix_cache_evictions;
    metrics->prefix_cache_partial_hits = source.prefix_cache_partial_hits;
    metrics->prefix_reused_tokens = source.prefix_reused_tokens;
    metrics->prefix_cow_pages = source.prefix_cow_pages;
    metrics->direct_paged_prefill_tokens = source.direct_paged_prefill_tokens;
    metrics->segmented_paged_decode_steps = source.segmented_paged_decode_steps;
    metrics->segmented_paged_decode_tokens = source.segmented_paged_decode_tokens;
    metrics->prefix_radix_nodes = source.prefix_radix_nodes;
    metrics->prefix_radix_lookups = source.prefix_radix_lookups;
    metrics->prefix_cow_bytes_copied = source.prefix_cow_bytes_copied;
    metrics->prefix_cow_bytes_saved = source.prefix_cow_bytes_saved;
    return LFM25_STATUS_OK;
}

} // extern "C"
