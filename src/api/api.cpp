#include "celeg/api.h"

#include "celeg/backend/cpu/concurrent.hpp"
#include "celeg/backend/cpu/model.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/backend/cpu/topology.hpp"
#include "celeg/serve/cpu_inference_service.hpp"
#include "celeg/text/tokenizer.hpp"
#ifdef CELEG_API_WITH_CUDA
#include "celeg/backend/cuda/cuda_inference_service.hpp"
#endif

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct celeg_model {
    celeg_backend backend = CELEG_BACKEND_CPU;
    std::unique_ptr<celeg::CpuModel> cpu;
    std::string error;
};
struct celeg_engine {
    std::unique_ptr<celeg::serve::ServiceBundle> service;
    std::string error;
};
struct celeg_tokenizer { std::unique_ptr<celeg::BpeTokenizer> value; std::string error; };

namespace {
thread_local std::string global_error;

template <typename Handle, typename Function>
celeg_status protect(Handle* handle, Function&& function) noexcept {
    if (!handle) return CELEG_STATUS_INVALID_ARGUMENT;
    try {
        function();
        handle->error.clear();
        return CELEG_STATUS_OK;
    } catch (const std::length_error& error) {
        handle->error = error.what();
        return CELEG_STATUS_BUFFER_TOO_SMALL;
    } catch (const std::out_of_range& error) {
        handle->error = error.what();
        return CELEG_STATUS_NOT_FOUND;
    } catch (const std::invalid_argument& error) {
        handle->error = error.what();
        return CELEG_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        handle->error = error.what();
        return CELEG_STATUS_CELEG_ERROR;
    } catch (...) {
        handle->error = "unknown runtime error";
        return CELEG_STATUS_CELEG_ERROR;
    }
}

void require_size(uint32_t actual, size_t expected, const char* name) {
    if (actual < expected) throw std::invalid_argument(std::string(name) + " struct is too small");
}

celeg::GenerationConfig generation(const celeg_generation_options& source) {
    celeg::GenerationConfig result;
    result.temperature = source.temperature;
    result.top_k = source.top_k;
    result.top_p = source.top_p;
    result.repetition_penalty = source.repetition_penalty;
    result.seed = source.seed;
    result.validate();
    return result;
}

celeg::CpuIsa cpu_isa(int value) {
    switch (value) {
        case 0: return celeg::CpuIsa::Auto; case 1: return celeg::CpuIsa::Scalar;
        case 2: return celeg::CpuIsa::Avx2; case 3: return celeg::CpuIsa::AvxVnni;
        case 4: return celeg::CpuIsa::Avx512Vnni; case 5: return celeg::CpuIsa::AmxInt8;
        case 6: return celeg::CpuIsa::Neon; case 7: return celeg::CpuIsa::DotProd;
        case 8: return celeg::CpuIsa::I8mm; case 9: return celeg::CpuIsa::Sve2;
        case 10: return celeg::CpuIsa::Sme2;
        default: throw std::invalid_argument("invalid CPU ISA");
    }
}

celeg::CpuAffinityPolicy cpu_affinity(int value) {
    switch (value) {
        case 0: return celeg::CpuAffinityPolicy::None;
        case 1: return celeg::CpuAffinityPolicy::Compact;
        case 2: return celeg::CpuAffinityPolicy::Scatter;
        default: throw std::invalid_argument("invalid CPU affinity policy");
    }
}

celeg::CpuKvCacheMode cpu_kv_cache_mode(int value) {
    switch (value) {
        case 0: return celeg::CpuKvCacheMode::Fp32;
        case 1: return celeg::CpuKvCacheMode::Bf16;
        default: throw std::invalid_argument("invalid CPU KV cache mode");
    }
}

celeg::CpuNumaMode cpu_numa_mode(int value) {
    switch (value) {
        case 0: return celeg::CpuNumaMode::Disabled;
        case 1: return celeg::CpuNumaMode::Local;
        case 2: return celeg::CpuNumaMode::ReplicateWeights;
        default: throw std::invalid_argument("invalid CPU NUMA mode");
    }
}

celeg::CpuModelOptions cpu_options(const celeg_cpu_model_config& input) {
    if (input.q4_group_size != 32 && input.q4_group_size != 64) {
        throw std::invalid_argument("CPU Q4 group size must be 32 or 64");
    }
    celeg::CpuModelOptions result;
    result.isa = cpu_isa(input.isa);
    result.threads = input.threads > 0 ? static_cast<size_t>(input.threads) : 0;
    result.weight_format = input.q4_group_size == 64 ? celeg::CpuWeightFormat::Q4Group64 : celeg::CpuWeightFormat::Q4Group32;
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

celeg::CpuModelOptions cpu_options(const celeg_cpu_model_options& source) {
    return cpu_options(source.cpu);
}

celeg::CpuModelOptions cpu_options(const celeg_engine_model_options& source) {
    if (source.backend != CELEG_BACKEND_CPU) {
        throw std::invalid_argument("CPU options require CPU backend");
    }
    return cpu_options(source.backend_options.cpu);
}

celeg::CpuConcurrentEngineOptions cpu_engine_options(const celeg_engine_options& source) {
    const auto& input = source.backend_options.cpu;
    celeg::CpuConcurrentEngineOptions result;
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

celeg_request_status status(celeg::serve::RequestStatus source) {
    switch (source) {
        case celeg::serve::RequestStatus::Queued: return CELEG_REQUEST_QUEUED;
        case celeg::serve::RequestStatus::Prefill: return CELEG_REQUEST_PREFILLING;
        case celeg::serve::RequestStatus::Decoding: return CELEG_REQUEST_DECODING;
        case celeg::serve::RequestStatus::Finished: return CELEG_REQUEST_COMPLETED;
        case celeg::serve::RequestStatus::Cancelled: return CELEG_REQUEST_CANCELLED;
        case celeg::serve::RequestStatus::Failed: return CELEG_REQUEST_FAILED;
    }
    return CELEG_REQUEST_FAILED;
}

#ifdef CELEG_API_WITH_CUDA
celeg::WeightMode cuda_weight_mode(int value) {
    switch (value) {
        case 0: return celeg::WeightMode::Bf16;
        case 1: return celeg::WeightMode::Int8;
        case 2: return celeg::WeightMode::Int4;
        default: throw std::invalid_argument("invalid CUDA weight mode");
    }
}

celeg::KvCacheMode cuda_kv_cache_mode(int value) {
    switch (value) {
        case 0: return celeg::KvCacheMode::Bf16;
        case 1: return celeg::KvCacheMode::Int8;
        default: throw std::invalid_argument("invalid CUDA KV cache mode");
    }
}

celeg::GemmBackend cuda_gemm_backend(int value) {
    switch (value) {
        case 0: return celeg::GemmBackend::Cublas;
        case 1: return celeg::GemmBackend::CublasLt;
        default: throw std::invalid_argument("invalid CUDA GEMM backend");
    }
}

celeg::AttentionMode cuda_attention_mode(int value) {
    switch (value) {
        case 0: return celeg::AttentionMode::Single;
        case 1: return celeg::AttentionMode::Segmented;
        case 2: return celeg::AttentionMode::Auto;
        default: throw std::invalid_argument("invalid CUDA attention mode");
    }
}

celeg::CudaModelOptions cuda_options(const celeg_engine_model_options& source) {
    const auto& input = source.backend_options.cuda;
    if (source.backend != CELEG_BACKEND_CUDA) throw std::invalid_argument("CUDA options require CUDA backend");
    if (input.flags != 0) throw std::invalid_argument("CUDA model option flags are reserved and must be zero");
    celeg::CudaModelOptions result;
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

celeg::SchedulerPolicy cuda_scheduler_policy(int value) {
    switch (value) {
        case 0: return celeg::SchedulerPolicy::GuaranteedNoEvict;
        case 1: return celeg::SchedulerPolicy::MaxUtilization;
        default: throw std::invalid_argument("invalid CUDA scheduler policy");
    }
}

celeg::ConcurrentEngineOptions cuda_engine_options(const celeg_engine_options& source) {
    const auto& input = source.backend_options.cuda;
    celeg::ConcurrentEngineOptions result;
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
void celeg_cpu_model_options_init(celeg_cpu_model_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_context = 4096;
    options->generation.struct_size = sizeof(options->generation);
    options->generation.temperature = 0.1f; options->generation.top_k = 50;
    options->generation.top_p = 1.0f; options->generation.repetition_penalty = 1.05f; options->generation.seed = 1;
    options->cpu.q4_group_size = 32;
    options->cpu.use_pack_cache = 1;
    options->cpu.kv_cache_mode = 1;
    options->cpu.kv_page_tokens = 32;
    options->cpu.prefill_chunk_tokens = 256;
    options->cpu.prefill_chunk_threshold = 16;
    options->cpu.attention_parallel_threshold = 256;
    options->cpu.attention_page_tile = 4;
}
void celeg_engine_options_init(celeg_engine_options* options, celeg_backend backend) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options); options->backend = backend;
    options->model = {};
    options->model.struct_size = sizeof(options->model);
    options->model.backend = backend;
    options->model.max_context = 4096;
    options->model.generation.struct_size = sizeof(options->model.generation);
    options->model.generation.temperature = 0.1f;
    options->model.generation.top_k = 50;
    options->model.generation.top_p = 1.0f;
    options->model.generation.repetition_penalty = 1.05f;
    options->model.generation.seed = 1;
    if (backend == CELEG_BACKEND_CPU) {
        options->model.backend_options.cpu.q4_group_size = 32;
        options->model.backend_options.cpu.use_pack_cache = 1;
        options->model.backend_options.cpu.kv_cache_mode = 1;
        options->model.backend_options.cpu.kv_page_tokens = 32;
        options->model.backend_options.cpu.prefill_chunk_tokens = 256;
        options->model.backend_options.cpu.prefill_chunk_threshold = 16;
        options->model.backend_options.cpu.attention_parallel_threshold = 256;
        options->model.backend_options.cpu.attention_page_tile = 4;
    } else if (backend == CELEG_BACKEND_CUDA) {
        options->model.backend_options.cuda.weight_mode = 0;
        options->model.backend_options.cuda.kv_cache_mode = 0;
        options->model.backend_options.cuda.gemm_backend = 0;
        options->model.backend_options.cuda.attention_mode = 2;
        options->model.backend_options.cuda.attention_chunk_tokens = 32;
        options->model.backend_options.cuda.attention_auto_threshold = 1;
        options->model.backend_options.cuda.lt_workspace_mb = 64;
        options->model.backend_options.cuda.lt_heuristics = 8;
    }
    if (backend == CELEG_BACKEND_CPU) {
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
    } else if (backend == CELEG_BACKEND_CUDA) {
        options->backend_options.cuda.max_active_requests = 8;
        options->backend_options.cuda.max_batched_tokens = 512;
        options->backend_options.cuda.prefill_chunk_tokens = 256;
        options->backend_options.cuda.page_tokens = 16;
    }
}
void celeg_request_options_init(celeg_request_options* options) {
    if (!options) return;
    *options = {}; options->struct_size = sizeof(*options); options->max_new_tokens = 128; options->eos_token_id = 7;
    options->generation.struct_size = sizeof(options->generation); options->generation.temperature = 0.1f;
    options->generation.top_k = 50; options->generation.top_p = 1.0f; options->generation.repetition_penalty = 1.05f; options->generation.seed = 1;
}
const char* celeg_backend_capabilities(celeg_backend backend) {
    if (backend == CELEG_BACKEND_CPU) {
        global_error = celeg::detect_cpu_capabilities().summary();
        return global_error.c_str();
    }
#ifdef CELEG_API_WITH_CUDA
    global_error = "CUDA backend available for celeg_engine_*; celeg_model_* remains CPU-only";
    return global_error.c_str();
#else
    return "CUDA backend unavailable in this build";
#endif
}

celeg_model* celeg_model_create(const char* path, const celeg_cpu_model_options* options) {
    if (!path || !*path || !options) { global_error = "model path and options are required"; return nullptr; }
    try {
        require_size(options->struct_size, sizeof(*options), "model options");
        require_size(options->generation.struct_size, sizeof(options->generation), "generation options");
        auto result = std::make_unique<celeg_model>();
        require_size(options->generation.struct_size, sizeof(options->generation), "generation options");
        result->cpu = std::make_unique<celeg::CpuModel>(path, options->max_context, cpu_options(*options), generation(options->generation));
        return result.release();
    } catch (const std::exception& error) { global_error = error.what(); return nullptr; }
}
void celeg_model_destroy(celeg_model* model) { delete model; }
celeg_status celeg_model_prefill(celeg_model* model, const int32_t* tokens, size_t count) {
    if (!tokens || count == 0) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { model->cpu->session().prefill(std::vector<int32_t>(tokens, tokens + count)); });
}
celeg_status celeg_model_decode(celeg_model* model, int32_t* token) {
    if (!token) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { *token = model->cpu->session().decode(); });
}
celeg_status celeg_model_copy_logits(celeg_model* model, float* output, size_t capacity, size_t* required) {
    if (!model || !required) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { const auto values = model->cpu->diagnostics().copy_logits(); *required = values.size(); if (!output || capacity < values.size()) throw std::length_error("logit output buffer is too small"); std::copy(values.begin(), values.end(), output); });
}
celeg_status celeg_model_get_metrics(celeg_model* model, celeg_metrics* metrics) {
    if (!model || !metrics || metrics->struct_size < sizeof(*metrics)) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { const auto value = model->cpu->diagnostics().runtime_metrics(); metrics->prefill_ms = value.last_prefill_ms; metrics->prefill_tokens = value.prefill_tokens; metrics->decode_ms = value.cumulative_decode_ms; metrics->decode_tokens = value.decoded_tokens; });
}
const char* celeg_model_last_error(const celeg_model* model) { return model ? model->error.c_str() : global_error.c_str(); }

celeg_engine* celeg_engine_create(const char* path, const celeg_engine_options* options) {
    if (!path || !*path || !options) { global_error = "engine path and options are required"; return nullptr; }
    try {
        require_size(options->struct_size, sizeof(*options), "engine options");
        require_size(options->model.struct_size, sizeof(options->model), "model options");
        if (options->backend != options->model.backend) {
            global_error = "engine backend must match model backend";
            return nullptr;
        }
        auto result = std::make_unique<celeg_engine>();
        if (options->backend == CELEG_BACKEND_CPU) {
            result->service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CpuInferenceService>(
                    path, options->model.max_context, cpu_options(options->model),
                    cpu_engine_options(*options)));
        } else if (options->backend == CELEG_BACKEND_CUDA) {
#ifdef CELEG_API_WITH_CUDA
            result->service = std::make_unique<celeg::serve::ServiceBundle>(
                std::make_unique<celeg::serve::CudaInferenceService>(
                    path, options->model.max_context, cuda_options(options->model),
                    cuda_engine_options(*options)));
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
void celeg_engine_destroy(celeg_engine* engine) { delete engine; }
celeg_status celeg_engine_submit(celeg_engine* engine, const int32_t* tokens, size_t count, const celeg_request_options* options, celeg_request_id* request_id) {
    if (!engine || !tokens || count == 0 || !options || !request_id) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(engine, [&] {
        require_size(options->struct_size, sizeof(*options), "request options");
        celeg::serve::GenerateRequest request;
        request.prompt_tokens.assign(tokens, tokens + count);
        request.max_output_tokens = options->max_new_tokens;
        request.eos_token_ids = {options->eos_token_id};
        request.priority = options->priority;
        request.generation = generation(options->generation);
        *request_id = engine->service->requests().submit(std::move(request));
    });
}
celeg_status celeg_engine_poll(celeg_engine* engine, celeg_request_id id, int32_t* output, size_t capacity, size_t* count, int* finished) {
    if (!engine || !output || capacity == 0 || !count || !finished) return CELEG_STATUS_INVALID_ARGUMENT;
    return protect(engine, [&] {
        const celeg::serve::GenerateEvent event = engine->service->requests().poll(id, capacity);
        std::copy(event.tokens.begin(), event.tokens.end(), output);
        *count = event.tokens.size();
        *finished = event.finished ? 1 : 0;
    });
}
celeg_status celeg_engine_status(celeg_engine* engine, celeg_request_id id, celeg_request_status* value) { if (!value) return CELEG_STATUS_INVALID_ARGUMENT; return protect(engine, [&] { *value = status(engine->service->requests().status(id)); }); }
celeg_status celeg_engine_cancel(celeg_engine* engine, celeg_request_id id) {
    return protect(engine, [&] {
        if (!engine->service->requests().cancel(id)) throw std::out_of_range("unknown request id");
    });
}
celeg_status celeg_engine_step(celeg_engine* engine, int* progressed) { if (!progressed) return CELEG_STATUS_INVALID_ARGUMENT; return protect(engine, [&] { *progressed = engine->service->scheduler().step() ? 1 : 0; }); }
const char* celeg_engine_last_error(const celeg_engine* engine) { return engine ? engine->error.c_str() : global_error.c_str(); }

celeg_tokenizer* celeg_tokenizer_create(const char* path) { if (!path || !*path) return nullptr; try { auto result = std::make_unique<celeg_tokenizer>(); result->value = std::make_unique<celeg::BpeTokenizer>(path); return result.release(); } catch (const std::exception& error) { global_error = error.what(); return nullptr; } }
void celeg_tokenizer_destroy(celeg_tokenizer* tokenizer) { delete tokenizer; }
celeg_status celeg_tokenizer_encode(celeg_tokenizer* tokenizer, const char* text, int add_bos, int32_t* output, size_t capacity, size_t* required) { if (!tokenizer || !text || !required) return CELEG_STATUS_INVALID_ARGUMENT; return protect(tokenizer, [&] { const auto values = tokenizer->value->encode(text, add_bos != 0); *required = values.size(); if (!output || capacity < values.size()) throw std::length_error("token output buffer is too small"); std::copy(values.begin(), values.end(), output); }); }
celeg_status celeg_tokenizer_decode(celeg_tokenizer* tokenizer, const int32_t* tokens, size_t count, int skip_special, char* output, size_t capacity, size_t* required) { if (!tokenizer || (!tokens && count) || !required) return CELEG_STATUS_INVALID_ARGUMENT; return protect(tokenizer, [&] { const std::vector<int32_t> values = count == 0 ? std::vector<int32_t>{} : std::vector<int32_t>(tokens, tokens + count); const auto text = tokenizer->value->decode(values, skip_special != 0); *required = text.size() + 1; if (!output || capacity < *required) throw std::length_error("text output buffer is too small"); std::memcpy(output, text.c_str(), *required); }); }
const char* celeg_tokenizer_last_error(const celeg_tokenizer* tokenizer) { return tokenizer ? tokenizer->error.c_str() : global_error.c_str(); }
} // extern "C"
