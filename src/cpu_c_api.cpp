#include "lfm/cpu_c_api.h"
#include "lfm/cpu_model.hpp"
#include "lfm/cpu_concurrent.hpp"
#include "lfm/cpu_topology.hpp"
#include "lfm/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct lfm25_cpu_model {
    std::unique_ptr<lfm::CpuModel> value;
    std::string error;
    std::string description;
    std::string pack_path;
};
struct lfm25_cpu_engine {
    std::unique_ptr<lfm::CpuConcurrentEngine> value;
    std::string error;
    std::string description;
};
struct lfm25_cpu_tokenizer {
    std::unique_ptr<lfm::BpeTokenizer> value;
    std::string error;
};

namespace {
thread_local std::string global_text;

lfm::CpuIsa convert_isa(int value) {
    switch (value) {
        case LFM25_CPU_ISA_AUTO: return lfm::CpuIsa::Auto;
        case LFM25_CPU_ISA_SCALAR: return lfm::CpuIsa::Scalar;
        case LFM25_CPU_ISA_AVX2: return lfm::CpuIsa::Avx2;
        case LFM25_CPU_ISA_AVX_VNNI: return lfm::CpuIsa::AvxVnni;
        case LFM25_CPU_ISA_AVX512_VNNI: return lfm::CpuIsa::Avx512Vnni;
        case LFM25_CPU_ISA_AMX_INT8: return lfm::CpuIsa::AmxInt8;
        case LFM25_CPU_ISA_NEON: return lfm::CpuIsa::Neon;
        case LFM25_CPU_ISA_DOTPROD: return lfm::CpuIsa::DotProd;
        case LFM25_CPU_ISA_I8MM: return lfm::CpuIsa::I8mm;
        case LFM25_CPU_ISA_SVE2: return lfm::CpuIsa::Sve2;
        case LFM25_CPU_ISA_SME2: return lfm::CpuIsa::Sme2;
        default: throw std::invalid_argument("invalid CPU ISA enum");
    }
}

lfm::CpuKvCacheMode convert_kv_mode(int value) {
    switch (value) {
        case LFM25_CPU_KV_FP32: return lfm::CpuKvCacheMode::Fp32;
        case LFM25_CPU_KV_BF16: return lfm::CpuKvCacheMode::Bf16;
        default: throw std::invalid_argument("invalid CPU KV cache mode");
    }
}

lfm::CpuNumaMode convert_numa_mode(int value) {
    switch (value) {
        case LFM25_CPU_NUMA_DISABLED: return lfm::CpuNumaMode::Disabled;
        case LFM25_CPU_NUMA_LOCAL: return lfm::CpuNumaMode::Local;
        case LFM25_CPU_NUMA_REPLICATE_WEIGHTS:
            return lfm::CpuNumaMode::ReplicateWeights;
        default: throw std::invalid_argument("invalid CPU NUMA mode");
    }
}

lfm::GenerationConfig convert_generation(
    const lfm25_cpu_generation_options_v1& source) {
    lfm::GenerationConfig generation;
    generation.temperature = source.temperature;
    generation.top_k = source.top_k;
    generation.top_p = source.top_p;
    generation.repetition_penalty = source.repetition_penalty;
    generation.seed = source.seed;
    generation.validate();
    return generation;
}

int convert_request_status(lfm::CpuRequestStatus status) {
    switch (status) {
        case lfm::CpuRequestStatus::Queued: return LFM25_CPU_REQUEST_QUEUED;
        case lfm::CpuRequestStatus::Prefilling: return LFM25_CPU_REQUEST_PREFILLING;
        case lfm::CpuRequestStatus::Decoding: return LFM25_CPU_REQUEST_DECODING;
        case lfm::CpuRequestStatus::Completed: return LFM25_CPU_REQUEST_COMPLETED;
        case lfm::CpuRequestStatus::Cancelled: return LFM25_CPU_REQUEST_CANCELLED;
        case lfm::CpuRequestStatus::Failed: return LFM25_CPU_REQUEST_FAILED;
    }
    return LFM25_CPU_REQUEST_FAILED;
}

lfm::CpuAffinityPolicy convert_affinity(int value) {
    switch (value) {
        case LFM25_CPU_AFFINITY_NONE: return lfm::CpuAffinityPolicy::None;
        case LFM25_CPU_AFFINITY_COMPACT: return lfm::CpuAffinityPolicy::Compact;
        case LFM25_CPU_AFFINITY_SCATTER: return lfm::CpuAffinityPolicy::Scatter;
        default: throw std::invalid_argument("invalid CPU affinity enum");
    }
}

lfm::CpuModelOptions convert_model_options(const lfm25_cpu_model_options_v6& source) {
    lfm::CpuModelOptions options;
    options.isa = convert_isa(source.isa);
    options.threads = source.threads > 0 ? static_cast<size_t>(source.threads) : 0;
    options.affinity = convert_affinity(source.affinity);
    options.kv_cache_mode = convert_kv_mode(source.kv_cache_mode);
    options.kv_page_tokens = source.kv_page_tokens;
    options.prefill_chunk_tokens = source.prefill_chunk_tokens;
    options.prefill_chunk_threshold = source.prefill_chunk_threshold;
    options.attention_parallel_threshold = source.attention_parallel_threshold;
    options.attention_page_tile = source.attention_page_tile;
    options.numa_mode = convert_numa_mode(source.numa_mode);
    options.weight_format = source.q4_group_size == 64
        ? lfm::CpuWeightFormat::Q4Group64 : lfm::CpuWeightFormat::Q4Group32;
    if (source.q4_group_size != 32 && source.q4_group_size != 64) {
        throw std::invalid_argument("CPU Q4 group size must be 32 or 64");
    }
    options.use_pack_cache = source.use_pack_cache != 0;
    if (source.pack_cache_directory) {
        options.pack_cache_directory = source.pack_cache_directory;
    }
    return options;
}

template <typename Handle, typename Function>
lfm25_cpu_status protect(Handle* handle, Function&& function) {
    if (!handle) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        function();
        handle->error.clear();
        return LFM25_CPU_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        handle->error = error.what();
        return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        handle->error = error.what();
        return LFM25_CPU_STATUS_RUNTIME_ERROR;
    }
}
}

extern "C" {

uint32_t lfm25_cpu_api_version(void) { return LFM25_CPU_C_API_VERSION; }

void lfm25_cpu_model_options_init(lfm25_cpu_model_options_v6* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_context = 4096;
    options->threads = 0;
    options->isa = LFM25_CPU_ISA_AUTO;
    options->q4_group_size = 32;
    options->use_pack_cache = 1;
    options->affinity = LFM25_CPU_AFFINITY_NONE;
    options->kv_cache_mode = LFM25_CPU_KV_BF16;
    options->kv_page_tokens = 32;
    options->prefill_chunk_tokens = 256;
    options->prefill_chunk_threshold = 16;
    options->attention_parallel_threshold = 256;
    options->attention_page_tile = 4;
    options->numa_mode = LFM25_CPU_NUMA_DISABLED;
}

void lfm25_cpu_generation_options_init(lfm25_cpu_generation_options_v1* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->temperature = 0.1f;
    options->top_k = 50;
    options->top_p = 1.0f;
    options->repetition_penalty = 1.05f;
    options->seed = 1;
}

void lfm25_cpu_engine_options_init(lfm25_cpu_engine_options_v3* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_active_requests = 16;
    options->max_batched_tokens = 256;
    options->max_prefill_batch = 16;
    options->max_decode_batch = 16;
    options->decode_first = 1;
    options->long_prefill_chunk_tokens = 256;
    options->long_prefill_threshold = 32;
    options->prefix_cache = 1;
    options->prefix_cache_max_entries = 256;
    options->prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
}

void lfm25_cpu_request_options_init(lfm25_cpu_request_options_v1* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_new_tokens = 128;
    options->eos_token_id = 7;
    options->priority = 0;
    lfm25_cpu_generation_options_init(&options->generation);
}

const char* lfm25_cpu_detected_isa(void) {
    global_text = lfm::cpu_isa_name(lfm::detect_cpu_capabilities().best_isa());
    return global_text.c_str();
}
const char* lfm25_cpu_capabilities(void) {
    global_text = lfm::detect_cpu_capabilities().summary();
    return global_text.c_str();
}
const char* lfm25_cpu_topology(void) {
    global_text = lfm::detect_cpu_topology().summary() + " " +
        lfm::detect_cpu_numa_topology().summary();
    return global_text.c_str();
}

lfm25_cpu_model* lfm25_cpu_model_create(
    const char* path, const lfm25_cpu_model_options_v6* input_options,
    const lfm25_cpu_generation_options_v1* input_generation) {
    lfm25_cpu_model_options_v6 defaults;
    lfm25_cpu_model_options_init(&defaults);
    const auto& source = input_options ? *input_options : defaults;
    if (input_options && input_options->struct_size < sizeof(lfm25_cpu_model_options_v6)) {
        global_text = "CPU options struct is too small";
        return nullptr;
    }
    lfm25_cpu_generation_options_v1 gen_defaults;
    lfm25_cpu_generation_options_init(&gen_defaults);
    const auto& gen_source = input_generation ? *input_generation : gen_defaults;
    if (input_generation && input_generation->struct_size < sizeof(lfm25_cpu_generation_options_v1)) {
        global_text = "CPU generation struct is too small";
        return nullptr;
    }
    auto handle = std::make_unique<lfm25_cpu_model>();
    try {
        if (!path || !*path) throw std::invalid_argument("safetensors_path is required");
        lfm::CpuModelOptions options = convert_model_options(source);
        lfm::GenerationConfig generation = convert_generation(gen_source);
        handle->value = std::make_unique<lfm::CpuModel>(
            path, source.max_context, options, generation);
        handle->description = handle->value->backend_description();
        handle->pack_path = handle->value->pack_path().string();
        return handle.release();
    } catch (const std::exception& error) {
        global_text = error.what();
        return nullptr;
    }
}

void lfm25_cpu_model_destroy(lfm25_cpu_model* model) { delete model; }

lfm25_cpu_status lfm25_cpu_model_prefill(lfm25_cpu_model* model,
                                         const int32_t* tokens, size_t count) {
    if (!tokens || count == 0) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { model->value->prefill(std::vector<int32_t>(tokens, tokens + count)); });
}
lfm25_cpu_status lfm25_cpu_model_decode(lfm25_cpu_model* model, int32_t* token) {
    if (!token) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] { *token = model->value->decode(); });
}
lfm25_cpu_status lfm25_cpu_model_copy_logits(lfm25_cpu_model* model, float* output,
                                              size_t capacity, size_t* required) {
    if (!model || !required) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        const std::vector<float> logits = model->value->copy_logits();
        *required = logits.size();
        if (!output || capacity < logits.size()) return LFM25_CPU_STATUS_BUFFER_TOO_SMALL;
        std::copy(logits.begin(), logits.end(), output);
        model->error.clear();
        return LFM25_CPU_STATUS_OK;
    } catch (const std::exception& error) {
        model->error = error.what();
        return LFM25_CPU_STATUS_RUNTIME_ERROR;
    }
}
lfm25_cpu_status lfm25_cpu_model_get_metrics(lfm25_cpu_model* model,
                                              lfm25_cpu_runtime_metrics_v1* out) {
    if (!out || out->struct_size < sizeof(*out)) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] {
        const auto m = model->value->runtime_metrics();
        out->prefill_ms = m.last_prefill_ms;
        out->prefill_tokens = m.prefill_tokens;
        out->prefill_tokens_per_second = m.prefill_tokens_per_second();
        out->decode_ms = m.cumulative_decode_ms;
        out->decode_tokens = m.decoded_tokens;
        out->decode_tokens_per_second = m.decode_tokens_per_second();
    });
}
lfm25_cpu_status lfm25_cpu_model_get_memory_stats(lfm25_cpu_model* model,
                                                   lfm25_cpu_memory_stats_v2* out) {
    if (!out || out->struct_size < sizeof(*out)) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    return protect(model, [&] {
        const auto s = model->value->memory_stats();
        out->weights = s.weights; out->kv_cache = s.kv_cache;
        out->conv_state = s.conv_state; out->activations = s.activations;
        out->total = s.total();
        out->kv_pages_used = s.kv_pages_used;
        out->kv_pages_total = s.kv_pages_total;
    });
}
lfm25_cpu_status lfm25_cpu_model_vocab_size(const lfm25_cpu_model* model,
                                             int32_t* vocab_size) {
    if (!model || !model->value || !vocab_size) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    *vocab_size = model->value->vocab_size();
    return LFM25_CPU_STATUS_OK;
}
const char* lfm25_cpu_model_backend_description(lfm25_cpu_model* model) {
    return model ? model->description.c_str() : "";
}
const char* lfm25_cpu_model_pack_path(lfm25_cpu_model* model) {
    return model ? model->pack_path.c_str() : "";
}
const char* lfm25_cpu_model_last_error(lfm25_cpu_model* model) {
    return model ? model->error.c_str() : global_text.c_str();
}

lfm25_cpu_engine* lfm25_cpu_engine_create(
    const char* path,
    const lfm25_cpu_model_options_v6* input_model,
    const lfm25_cpu_engine_options_v3* input_engine) {
    auto handle = std::make_unique<lfm25_cpu_engine>();
    try {
        if (!path || !*path) throw std::invalid_argument("safetensors_path is required");
        lfm25_cpu_model_options_v6 model_defaults;
        lfm25_cpu_model_options_init(&model_defaults);
        lfm25_cpu_engine_options_v3 engine_defaults;
        lfm25_cpu_engine_options_init(&engine_defaults);
        const auto& model_source = input_model ? *input_model : model_defaults;
        const auto& engine_source = input_engine ? *input_engine : engine_defaults;
        if (input_model && input_model->struct_size < sizeof(lfm25_cpu_model_options_v6)) {
            throw std::invalid_argument("CPU model options struct is too small");
        }
        if (input_engine && input_engine->struct_size < sizeof(lfm25_cpu_engine_options_v3)) {
            throw std::invalid_argument("CPU engine options struct is too small");
        }
        lfm::CpuModelOptions model_options = convert_model_options(model_source);
        lfm::CpuConcurrentEngineOptions engine_options;
        engine_options.max_active_requests = engine_source.max_active_requests;
        engine_options.max_batched_tokens = engine_source.max_batched_tokens;
        engine_options.max_prefill_batch = engine_source.max_prefill_batch;
        engine_options.max_decode_batch = engine_source.max_decode_batch;
        engine_options.decode_first = engine_source.decode_first != 0;
        engine_options.long_prefill_chunk_tokens = engine_source.long_prefill_chunk_tokens;
        engine_options.long_prefill_threshold = engine_source.long_prefill_threshold;
        engine_options.prefix_cache = engine_source.prefix_cache != 0;
        engine_options.prefix_cache_max_entries = engine_source.prefix_cache_max_entries;
        engine_options.prefix_cache_max_bytes = engine_source.prefix_cache_max_bytes;
        handle->value = std::make_unique<lfm::CpuConcurrentEngine>(
            path, model_source.max_context, model_options, engine_options);
        handle->description = handle->value->backend_description();
        return handle.release();
    } catch (const std::exception& error) {
        global_text = error.what();
        return nullptr;
    }
}
void lfm25_cpu_engine_destroy(lfm25_cpu_engine* engine) { delete engine; }
lfm25_cpu_status lfm25_cpu_engine_submit(
    lfm25_cpu_engine* engine, const int32_t* tokens, size_t count,
    const lfm25_cpu_request_options_v1* input_options,
    lfm25_cpu_request_id* request_id) {
    if (!engine || !tokens || count == 0 || !request_id) {
        return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    }
    return protect(engine, [&] {
        lfm25_cpu_request_options_v1 defaults;
        lfm25_cpu_request_options_init(&defaults);
        const auto& source = input_options ? *input_options : defaults;
        if (input_options && input_options->struct_size < sizeof(lfm25_cpu_request_options_v1)) {
            throw std::invalid_argument("CPU request options struct is too small");
        }
        if (source.generation.struct_size < sizeof(lfm25_cpu_generation_options_v1)) {
            throw std::invalid_argument("CPU request generation struct is too small");
        }
        lfm::CpuRequestOptions options;
        options.max_new_tokens = source.max_new_tokens;
        options.eos_token_id = source.eos_token_id;
        options.priority = source.priority;
        options.generation = convert_generation(source.generation);
        *request_id = engine->value->submit(
            std::vector<int32_t>(tokens, tokens + count), options);
    });
}
lfm25_cpu_status lfm25_cpu_engine_poll(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id,
    int32_t* output, size_t capacity, size_t* token_count, int* finished) {
    if (!engine || !output || capacity == 0 || !token_count || !finished) {
        return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    }
    try {
        bool done = false;
        const std::vector<int32_t> tokens = engine->value->poll(
            request_id, capacity, &done);
        *token_count = tokens.size();
        *finished = done ? 1 : 0;
        std::copy(tokens.begin(), tokens.end(), output);
        engine->error.clear();
        return LFM25_CPU_STATUS_OK;
    } catch (const std::out_of_range& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_NOT_FOUND;
    } catch (const std::invalid_argument& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_RUNTIME_ERROR;
    }
}
lfm25_cpu_status lfm25_cpu_engine_request_status(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id,
    int32_t* status) {
    if (!engine || !status) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        *status = convert_request_status(engine->value->status(request_id));
        engine->error.clear();
        return LFM25_CPU_STATUS_OK;
    } catch (const std::out_of_range& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_NOT_FOUND;
    } catch (const std::exception& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_RUNTIME_ERROR;
    }
}
lfm25_cpu_status lfm25_cpu_engine_cancel(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id) {
    if (!engine) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        engine->value->cancel(request_id);
        engine->error.clear();
        return LFM25_CPU_STATUS_OK;
    } catch (const std::out_of_range& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_NOT_FOUND;
    } catch (const std::exception& error) {
        engine->error = error.what();
        return LFM25_CPU_STATUS_RUNTIME_ERROR;
    }
}
lfm25_cpu_status lfm25_cpu_engine_step(lfm25_cpu_engine* engine, int* progressed) {
    if (!engine || !progressed) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    return protect(engine, [&] { *progressed = engine->value->step() ? 1 : 0; });
}
lfm25_cpu_status lfm25_cpu_engine_start(lfm25_cpu_engine* engine) {
    return protect(engine, [&] { engine->value->start(); });
}
lfm25_cpu_status lfm25_cpu_engine_stop(lfm25_cpu_engine* engine) {
    return protect(engine, [&] { engine->value->stop(); });
}
lfm25_cpu_status lfm25_cpu_engine_get_metrics(
    lfm25_cpu_engine* engine, lfm25_cpu_engine_metrics_v3* out) {
    if (!engine || !out || out->struct_size < sizeof(*out)) {
        return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    }
    return protect(engine, [&] {
        const lfm::CpuConcurrentMetrics m = engine->value->metrics();
        out->submitted_requests = m.submitted_requests;
        out->completed_requests = m.completed_requests;
        out->cancelled_requests = m.cancelled_requests;
        out->failed_requests = m.failed_requests;
        out->active_requests = m.active_requests;
        out->queued_requests = m.queued_requests;
        out->prefill_tokens = m.prefill_tokens;
        out->decode_tokens = m.decode_tokens;
        out->ragged_prefill_steps = m.ragged_prefill_steps;
        out->packed_decode_steps = m.packed_decode_steps;
        out->maximum_prefill_batch = m.maximum_prefill_batch;
        out->maximum_decode_batch = m.maximum_decode_batch;
        out->prefill_tokens_per_second = m.prefill_tokens_per_second();
        out->decode_tokens_per_second = m.decode_tokens_per_second();
        out->average_ttft_ms = m.average_ttft_ms();
        out->average_itl_ms = m.average_itl_ms();
        out->scheduler_ms = m.cumulative_scheduler_ms;
        out->chunked_prefill_steps = m.chunked_prefill_steps;
        out->chunked_prefill_tokens = m.chunked_prefill_tokens;
        out->maximum_prefill_chunk = m.maximum_prefill_chunk;
        out->prefix_cache_hits = m.prefix_cache_hits;
        out->prefix_cache_misses = m.prefix_cache_misses;
        out->prefix_cache_partial_hits = m.prefix_cache_partial_hits;
        out->prefix_cache_inserts = m.prefix_cache_inserts;
        out->prefix_cache_evictions = m.prefix_cache_evictions;
        out->prefix_reused_tokens = m.prefix_reused_tokens;
        out->prefix_cow_pages = m.prefix_cow_pages;
        out->prefix_cow_bytes = m.prefix_cow_bytes;
        out->attention_parallel_calls = m.attention_parallel_calls;
    });
}
const char* lfm25_cpu_engine_backend_description(lfm25_cpu_engine* engine) {
    return engine ? engine->description.c_str() : "";
}
const char* lfm25_cpu_engine_last_error(lfm25_cpu_engine* engine) {
    if (!engine) return global_text.c_str();
    if (!engine->error.empty()) return engine->error.c_str();
    engine->error = engine->value->last_error();
    return engine->error.c_str();
}

lfm25_cpu_tokenizer* lfm25_cpu_tokenizer_create(const char* path) {
    if (!path || !*path) return nullptr;
    auto handle = std::make_unique<lfm25_cpu_tokenizer>();
    try { handle->value = std::make_unique<lfm::BpeTokenizer>(path); return handle.release(); }
    catch (const std::exception& error) { global_text = error.what(); return nullptr; }
}
void lfm25_cpu_tokenizer_destroy(lfm25_cpu_tokenizer* tokenizer) { delete tokenizer; }
lfm25_cpu_status lfm25_cpu_tokenizer_encode(
    lfm25_cpu_tokenizer* tokenizer, const char* text, int add_bos,
    int32_t* output, size_t capacity, size_t* required) {
    if (!tokenizer || !text || !required) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        const std::vector<int32_t> tokens = tokenizer->value->encode(text, add_bos != 0);
        *required = tokens.size();
        if (!output || capacity < tokens.size()) return LFM25_CPU_STATUS_BUFFER_TOO_SMALL;
        std::copy(tokens.begin(), tokens.end(), output);
        tokenizer->error.clear(); return LFM25_CPU_STATUS_OK;
    } catch (const std::exception& error) { tokenizer->error = error.what(); return LFM25_CPU_STATUS_RUNTIME_ERROR; }
}
lfm25_cpu_status lfm25_cpu_tokenizer_decode(
    lfm25_cpu_tokenizer* tokenizer, const int32_t* tokens, size_t count,
    int skip_special, char* output, size_t capacity, size_t* required) {
    if (!tokenizer || (!tokens && count) || !required) return LFM25_CPU_STATUS_INVALID_ARGUMENT;
    try {
        std::vector<int32_t> token_vector;
        if (count != 0) token_vector.assign(tokens, tokens + count);
        const std::string text = tokenizer->value->decode(
            token_vector, skip_special != 0);
        *required = text.size() + 1;
        if (!output || capacity < text.size() + 1) return LFM25_CPU_STATUS_BUFFER_TOO_SMALL;
        std::memcpy(output, text.c_str(), text.size() + 1);
        tokenizer->error.clear(); return LFM25_CPU_STATUS_OK;
    } catch (const std::exception& error) { tokenizer->error = error.what(); return LFM25_CPU_STATUS_RUNTIME_ERROR; }
}
const char* lfm25_cpu_tokenizer_last_error(lfm25_cpu_tokenizer* tokenizer) {
    return tokenizer ? tokenizer->error.c_str() : global_text.c_str();
}
}
