#include "lfm/api.h"

#include "lfm/backend/cpu/concurrent.hpp"
#include "lfm/backend/cpu/model.hpp"
#include "lfm/backend/cpu/numa.hpp"
#include "lfm/backend/cpu/topology.hpp"
#include "lfm/serve/cpu_inference_service.hpp"
#include "lfm/text/tokenizer.hpp"
#ifdef LFM25_API_WITH_CUDA
#include "lfm/serve/cuda_inference_service.hpp"
#endif

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct lfm25_model {
    lfm25_backend backend = LFM25_BACKEND_CPU;
    std::unique_ptr<lfm::CpuModel> cpu;
    std::string error;
};
struct lfm25_engine {
    std::unique_ptr<lfm::serve::IInferenceService> service;
    std::string error;
};
struct lfm25_tokenizer { std::unique_ptr<lfm::BpeTokenizer> value; std::string error; };

namespace {
thread_local std::string global_error;

template <typename Handle, typename Function>
lfm25_status protect(Handle* handle, Function&& function) noexcept {
    if (!handle) return LFM25_STATUS_INVALID_ARGUMENT;
    try {
        function();
        handle->error.clear();
        return LFM25_STATUS_OK;
    } catch (const std::length_error& error) {
        handle->error = error.what();
        return LFM25_STATUS_BUFFER_TOO_SMALL;
    } catch (const std::out_of_range& error) {
        handle->error = error.what();
        return LFM25_STATUS_NOT_FOUND;
    } catch (const std::invalid_argument& error) {
        handle->error = error.what();
        return LFM25_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        handle->error = error.what();
        return LFM25_STATUS_RUNTIME_ERROR;
    } catch (...) {
        handle->error = "unknown runtime error";
        return LFM25_STATUS_RUNTIME_ERROR;
    }
}

void require_size(uint32_t actual, size_t expected, const char* name) {
    if (actual < expected) throw std::invalid_argument(std::string(name) + " struct is too small");
}

lfm::GenerationConfig generation(const lfm25_generation_options& source) {
    lfm::GenerationConfig result;
    result.temperature = source.temperature;
    result.top_k = source.top_k;
    result.top_p = source.top_p;
    result.repetition_penalty = source.repetition_penalty;
    result.seed = source.seed;
    result.validate();
    return result;
}

lfm::CpuIsa cpu_isa(int value) {
    switch (value) {
        case 0: return lfm::CpuIsa::Auto; case 1: return lfm::CpuIsa::Scalar;
        case 2: return lfm::CpuIsa::Avx2; case 3: return lfm::CpuIsa::AvxVnni;
        case 4: return lfm::CpuIsa::Avx512Vnni; case 5: return lfm::CpuIsa::AmxInt8;
        case 6: return lfm::CpuIsa::Neon; case 7: return lfm::CpuIsa::DotProd;
        case 8: return lfm::CpuIsa::I8mm; case 9: return lfm::CpuIsa::Sve2;
        case 10: return lfm::CpuIsa::Sme2;
        default: throw std::invalid_argument("invalid CPU ISA");
    }
}

lfm::CpuAffinityPolicy cpu_affinity(int value) {
    switch (value) {
        case 0: return lfm::CpuAffinityPolicy::None;
        case 1: return lfm::CpuAffinityPolicy::Compact;
        case 2: return lfm::CpuAffinityPolicy::Scatter;
        default: throw std::invalid_argument("invalid CPU affinity policy");
    }
}

lfm::CpuKvCacheMode cpu_kv_cache_mode(int value) {
    switch (value) {
        case 0: return lfm::CpuKvCacheMode::Fp32;
        case 1: return lfm::CpuKvCacheMode::Bf16;
        default: throw std::invalid_argument("invalid CPU KV cache mode");
    }
}

lfm::CpuNumaMode cpu_numa_mode(int value) {
    switch (value) {
        case 0: return lfm::CpuNumaMode::Disabled;
        case 1: return lfm::CpuNumaMode::Local;
        case 2: return lfm::CpuNumaMode::ReplicateWeights;
        default: throw std::invalid_argument("invalid CPU NUMA mode");
    }
}

lfm::CpuModelOptions cpu_options(const lfm25_model_options& source) {
    const auto& input = source.backend_options.cpu;
    if (source.backend != LFM25_BACKEND_CPU) throw std::invalid_argument("CPU options require CPU backend");
    if (input.q4_group_size != 32 && input.q4_group_size != 64) {
        throw std::invalid_argument("CPU Q4 group size must be 32 or 64");
    }
    lfm::CpuModelOptions result;
    result.isa = cpu_isa(input.isa);
    result.threads = input.threads > 0 ? static_cast<size_t>(input.threads) : 0;
    result.weight_format = input.q4_group_size == 64 ? lfm::CpuWeightFormat::Q4Group64 : lfm::CpuWeightFormat::Q4Group32;
    result.use_pack_cache = input.use_pack_cache != 0;
    if (input.pack_cache_directory) result.pack_cache_directory = input.pack_cache_directory;
    result.affinity = cpu_affinity(input.affinity);
    result.kv_cache_mode = cpu_kv_cache_mode(input.kv_cache_mode);
    result.kv_page_tokens = input.kv_page_tokens;
    result.prefill_chunk_tokens = input.prefill_chunk_tokens;
    result.prefill_chunk_threshold = input.prefill_chunk_threshold;
    result.attention_parallel_threshold = input.attention_parallel_threshold;
    result.attention_page_tile = input.attention_page_tile;
    result.numa_mode = cpu_numa_mode(input.numa_mode);
    return result;
}

lfm::CpuConcurrentEngineOptions cpu_engine_options(const lfm25_engine_options& source) {
    const auto& input = source.backend_options.cpu;
    lfm::CpuConcurrentEngineOptions result;
    result.max_active_requests = input.max_active_requests;
    result.max_batched_tokens = input.max_batched_tokens;
    result.max_prefill_batch = input.max_prefill_batch;
    result.max_decode_batch = input.max_decode_batch;
    result.decode_first = input.decode_first != 0;
    result.long_prefill_chunk_tokens = input.long_prefill_chunk_tokens;
    result.long_prefill_threshold = input.long_prefill_threshold;
    result.prefix_cache = input.prefix_cache != 0;
    result.prefix_cache_max_entries = input.prefix_cache_max_entries;
    result.prefix_cache_max_bytes = input.prefix_cache_max_bytes;
    return result;
}

lfm25_request_status status(lfm::serve::RequestStatus source) {
    switch (source) {
        case lfm::serve::RequestStatus::Queued: return LFM25_REQUEST_QUEUED;
        case lfm::serve::RequestStatus::Prefill: return LFM25_REQUEST_PREFILLING;
        case lfm::serve::RequestStatus::Decoding: return LFM25_REQUEST_DECODING;
        case lfm::serve::RequestStatus::Finished: return LFM25_REQUEST_COMPLETED;
        case lfm::serve::RequestStatus::Cancelled: return LFM25_REQUEST_CANCELLED;
        case lfm::serve::RequestStatus::Failed: return LFM25_REQUEST_FAILED;
    }
    return LFM25_REQUEST_FAILED;
}

#ifdef LFM25_API_WITH_CUDA
lfm::WeightMode cuda_weight_mode(int value) {
    switch (value) {
        case 0: return lfm::WeightMode::Bf16;
        case 1: return lfm::WeightMode::Int8;
        case 2: return lfm::WeightMode::Int4;
        default: throw std::invalid_argument("invalid CUDA weight mode");
    }
}

lfm::KvCacheMode cuda_kv_cache_mode(int value) {
    switch (value) {
        case 0: return lfm::KvCacheMode::Bf16;
        case 1: return lfm::KvCacheMode::Int8;
        default: throw std::invalid_argument("invalid CUDA KV cache mode");
    }
}

lfm::GemmBackend cuda_gemm_backend(int value) {
    switch (value) {
        case 0: return lfm::GemmBackend::Cublas;
        case 1: return lfm::GemmBackend::CublasLt;
        default: throw std::invalid_argument("invalid CUDA GEMM backend");
    }
}

lfm::AttentionMode cuda_attention_mode(int value) {
    switch (value) {
        case 0: return lfm::AttentionMode::Single;
        case 1: return lfm::AttentionMode::Segmented;
        case 2: return lfm::AttentionMode::Auto;
        default: throw std::invalid_argument("invalid CUDA attention mode");
    }
}

lfm::ModelOptions cuda_options(const lfm25_model_options& source) {
    const auto& input = source.backend_options.cuda;
    if (source.backend != LFM25_BACKEND_CUDA) throw std::invalid_argument("CUDA options require CUDA backend");
    if (input.flags != 0) throw std::invalid_argument("CUDA model option flags are reserved and must be zero");
    lfm::ModelOptions result;
    result.weight_mode = cuda_weight_mode(input.weight_mode);
    result.kv_cache_mode = cuda_kv_cache_mode(input.kv_cache_mode);
    result.gemm_backend = cuda_gemm_backend(input.gemm_backend);
    result.attention_mode = cuda_attention_mode(input.attention_mode);
    result.attention_chunk_tokens = input.attention_chunk_tokens;
    result.attention_auto_threshold = input.attention_auto_threshold;
    if (input.lt_workspace_mb > 0) {
        result.lt_workspace_bytes = static_cast<size_t>(input.lt_workspace_mb) * 1024ULL * 1024ULL;
    }
    result.lt_heuristics = input.lt_heuristics;
    return result;
}

lfm::SchedulerPolicy cuda_scheduler_policy(int value) {
    switch (value) {
        case 0: return lfm::SchedulerPolicy::GuaranteedNoEvict;
        case 1: return lfm::SchedulerPolicy::MaxUtilization;
        default: throw std::invalid_argument("invalid CUDA scheduler policy");
    }
}

lfm::ConcurrentEngineOptions cuda_engine_options(const lfm25_engine_options& source) {
    const auto& input = source.backend_options.cuda;
    lfm::ConcurrentEngineOptions result;
    result.max_active_requests = input.max_active_requests;
    result.max_batched_tokens = input.max_batched_tokens;
    result.prefill_chunk_tokens = input.prefill_chunk_tokens;
    result.page_tokens = input.page_tokens;
    result.logical_kv_pages = input.logical_kv_pages;
    result.scheduler_policy = cuda_scheduler_policy(input.scheduler_policy);
    return result;
}
#endif
} // namespace

extern "C" {
void lfm25_model_options_init(lfm25_model_options* options, lfm25_backend backend) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->backend = backend;
    options->max_context = 4096;
    options->generation.struct_size = sizeof(options->generation);
    options->generation.temperature = 0.1f; options->generation.top_k = 50;
    options->generation.top_p = 1.0f; options->generation.repetition_penalty = 1.05f; options->generation.seed = 1;
    if (backend == LFM25_BACKEND_CPU) {
        options->backend_options.cpu.q4_group_size = 32;
        options->backend_options.cpu.use_pack_cache = 1;
        options->backend_options.cpu.kv_cache_mode = 1;
        options->backend_options.cpu.kv_page_tokens = 32;
        options->backend_options.cpu.prefill_chunk_tokens = 256;
        options->backend_options.cpu.prefill_chunk_threshold = 16;
        options->backend_options.cpu.attention_parallel_threshold = 256;
        options->backend_options.cpu.attention_page_tile = 4;
    } else if (backend == LFM25_BACKEND_CUDA) {
        options->backend_options.cuda.weight_mode = 0;
        options->backend_options.cuda.kv_cache_mode = 0;
        options->backend_options.cuda.gemm_backend = 0;
        options->backend_options.cuda.attention_mode = 2;
        options->backend_options.cuda.attention_chunk_tokens = 256;
        options->backend_options.cuda.attention_auto_threshold = 4096;
        options->backend_options.cuda.lt_workspace_mb = 64;
        options->backend_options.cuda.lt_heuristics = 8;
    }
}
void lfm25_engine_options_init(lfm25_engine_options* options, lfm25_backend backend) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options); options->backend = backend;
    lfm25_model_options_init(&options->model, backend);
    if (backend == LFM25_BACKEND_CPU) {
        options->backend_options.cpu.max_active_requests = 16;
        options->backend_options.cpu.max_batched_tokens = 256;
        options->backend_options.cpu.max_prefill_batch = 16;
        options->backend_options.cpu.max_decode_batch = 16;
        options->backend_options.cpu.decode_first = 1;
        options->backend_options.cpu.long_prefill_chunk_tokens = 256;
        options->backend_options.cpu.long_prefill_threshold = 32;
        options->backend_options.cpu.prefix_cache = 1;
        options->backend_options.cpu.prefix_cache_max_entries = 256;
        options->backend_options.cpu.prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
    } else if (backend == LFM25_BACKEND_CUDA) {
        options->backend_options.cuda.max_active_requests = 8;
        options->backend_options.cuda.max_batched_tokens = 512;
        options->backend_options.cuda.prefill_chunk_tokens = 256;
        options->backend_options.cuda.page_tokens = 16;
    }
}
void lfm25_request_options_init(lfm25_request_options* options) {
    if (!options) return;
    *options = {}; options->struct_size = sizeof(*options); options->max_new_tokens = 128; options->eos_token_id = 7;
    options->generation.struct_size = sizeof(options->generation); options->generation.temperature = 0.1f;
    options->generation.top_k = 50; options->generation.top_p = 1.0f; options->generation.repetition_penalty = 1.05f; options->generation.seed = 1;
}
const char* lfm25_backend_capabilities(lfm25_backend backend) {
    if (backend == LFM25_BACKEND_CPU) {
        global_error = lfm::detect_cpu_capabilities().summary();
        return global_error.c_str();
    }
#ifdef LFM25_API_WITH_CUDA
    global_error = "CUDA backend available for lfm25_engine_*; lfm25_model_* remains CPU-only";
    return global_error.c_str();
#else
    return "CUDA backend unavailable in this build";
#endif
}

lfm25_model* lfm25_model_create(const char* path, const lfm25_model_options* options) {
    if (!path || !*path || !options) { global_error = "model path and options are required"; return nullptr; }
    try {
        require_size(options->struct_size, sizeof(*options), "model options");
        require_size(options->generation.struct_size, sizeof(options->generation), "generation options");
        if (options->backend != LFM25_BACKEND_CPU) { global_error = "requested backend is unavailable"; return nullptr; }
        auto result = std::make_unique<lfm25_model>();
        result->cpu = std::make_unique<lfm::CpuModel>(path, options->max_context, cpu_options(*options), generation(options->generation));
        return result.release();
    } catch (const std::exception& error) { global_error = error.what(); return nullptr; }
}
void lfm25_model_destroy(lfm25_model* model) { delete model; }
lfm25_status lfm25_model_prefill(lfm25_model* model, const int32_t* tokens, size_t count) {
    if (!tokens || count == 0) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { model->cpu->session().prefill(std::vector<int32_t>(tokens, tokens + count)); });
}
lfm25_status lfm25_model_decode(lfm25_model* model, int32_t* token) {
    if (!token) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { *token = model->cpu->session().decode(); });
}
lfm25_status lfm25_model_copy_logits(lfm25_model* model, float* output, size_t capacity, size_t* required) {
    if (!model || !required) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { const auto values = model->cpu->diagnostics().copy_logits(); *required = values.size(); if (!output || capacity < values.size()) throw std::length_error("logit output buffer is too small"); std::copy(values.begin(), values.end(), output); });
}
lfm25_status lfm25_model_get_metrics(lfm25_model* model, lfm25_runtime_metrics* metrics) {
    if (!model || !metrics || metrics->struct_size < sizeof(*metrics)) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { const auto value = model->cpu->diagnostics().runtime_metrics(); metrics->prefill_ms = value.last_prefill_ms; metrics->prefill_tokens = value.prefill_tokens; metrics->decode_ms = value.cumulative_decode_ms; metrics->decode_tokens = value.decoded_tokens; });
}
const char* lfm25_model_last_error(const lfm25_model* model) { return model ? model->error.c_str() : global_error.c_str(); }

lfm25_engine* lfm25_engine_create(const char* path, const lfm25_engine_options* options) {
    if (!path || !*path || !options) { global_error = "engine path and options are required"; return nullptr; }
    try {
        require_size(options->struct_size, sizeof(*options), "engine options");
        require_size(options->model.struct_size, sizeof(options->model), "model options");
        if (options->backend != options->model.backend) {
            global_error = "engine backend must match model backend";
            return nullptr;
        }
        auto result = std::make_unique<lfm25_engine>();
        if (options->backend == LFM25_BACKEND_CPU) {
            result->service = std::make_unique<lfm::serve::CpuInferenceService>(
                path, options->model.max_context, cpu_options(options->model),
                cpu_engine_options(*options));
        } else if (options->backend == LFM25_BACKEND_CUDA) {
#ifdef LFM25_API_WITH_CUDA
            result->service = std::make_unique<lfm::serve::CudaInferenceService>(
                path, options->model.max_context, cuda_options(options->model),
                cuda_engine_options(*options));
#else
            global_error = "CUDA backend is unavailable in this build";
            return nullptr;
#endif
        } else {
            global_error = "unknown backend";
            return nullptr;
        }
        return result.release();
    } catch (const std::exception& error) { global_error = error.what(); return nullptr; }
}
void lfm25_engine_destroy(lfm25_engine* engine) { delete engine; }
lfm25_status lfm25_engine_submit(lfm25_engine* engine, const int32_t* tokens, size_t count, const lfm25_request_options* options, lfm25_request_id* request_id) {
    if (!engine || !tokens || count == 0 || !options || !request_id) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(engine, [&] {
        require_size(options->struct_size, sizeof(*options), "request options");
        lfm::serve::GenerateRequest request;
        request.prompt_tokens.assign(tokens, tokens + count);
        request.max_output_tokens = options->max_new_tokens;
        request.eos_token_id = options->eos_token_id;
        request.priority = options->priority;
        request.generation = generation(options->generation);
        *request_id = engine->service->submit(std::move(request));
    });
}
lfm25_status lfm25_engine_poll(lfm25_engine* engine, lfm25_request_id id, int32_t* output, size_t capacity, size_t* count, int* finished) {
    if (!engine || !output || capacity == 0 || !count || !finished) return LFM25_STATUS_INVALID_ARGUMENT;
    return protect(engine, [&] {
        const lfm::serve::GenerateEvent event = engine->service->poll(id, capacity);
        std::copy(event.tokens.begin(), event.tokens.end(), output);
        *count = event.tokens.size();
        *finished = event.finished ? 1 : 0;
    });
}
lfm25_status lfm25_engine_status(lfm25_engine* engine, lfm25_request_id id, lfm25_request_status* value) { if (!value) return LFM25_STATUS_INVALID_ARGUMENT; return protect(engine, [&] { *value = status(engine->service->status(id)); }); }
lfm25_status lfm25_engine_cancel(lfm25_engine* engine, lfm25_request_id id) {
    return protect(engine, [&] {
        if (!engine->service->cancel(id)) throw std::out_of_range("unknown request id");
    });
}
lfm25_status lfm25_engine_step(lfm25_engine* engine, int* progressed) { if (!progressed) return LFM25_STATUS_INVALID_ARGUMENT; return protect(engine, [&] { *progressed = engine->service->step() ? 1 : 0; }); }
const char* lfm25_engine_last_error(const lfm25_engine* engine) { return engine ? engine->error.c_str() : global_error.c_str(); }

lfm25_tokenizer* lfm25_tokenizer_create(const char* path) { if (!path || !*path) return nullptr; try { auto result = std::make_unique<lfm25_tokenizer>(); result->value = std::make_unique<lfm::BpeTokenizer>(path); return result.release(); } catch (const std::exception& error) { global_error = error.what(); return nullptr; } }
void lfm25_tokenizer_destroy(lfm25_tokenizer* tokenizer) { delete tokenizer; }
lfm25_status lfm25_tokenizer_encode(lfm25_tokenizer* tokenizer, const char* text, int add_bos, int32_t* output, size_t capacity, size_t* required) { if (!tokenizer || !text || !required) return LFM25_STATUS_INVALID_ARGUMENT; return protect(tokenizer, [&] { const auto values = tokenizer->value->encode(text, add_bos != 0); *required = values.size(); if (!output || capacity < values.size()) throw std::length_error("token output buffer is too small"); std::copy(values.begin(), values.end(), output); }); }
lfm25_status lfm25_tokenizer_decode(lfm25_tokenizer* tokenizer, const int32_t* tokens, size_t count, int skip_special, char* output, size_t capacity, size_t* required) { if (!tokenizer || (!tokens && count) || !required) return LFM25_STATUS_INVALID_ARGUMENT; return protect(tokenizer, [&] { const std::vector<int32_t> values = count == 0 ? std::vector<int32_t>{} : std::vector<int32_t>(tokens, tokens + count); const auto text = tokenizer->value->decode(values, skip_special != 0); *required = text.size() + 1; if (!output || capacity < *required) throw std::length_error("text output buffer is too small"); std::memcpy(output, text.c_str(), *required); }); }
const char* lfm25_tokenizer_last_error(const lfm25_tokenizer* tokenizer) { return tokenizer ? tokenizer->error.c_str() : global_error.c_str(); }
} // extern "C"
