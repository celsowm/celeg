#include "lfm/api/cuda.h"
#include "lfm/model/model.hpp"
#include "lfm/text/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct lfm25_model {
    std::unique_ptr<lfm::LfmModel> engine;
    mutable std::string error;
};

struct lfm25_tokenizer {
    std::unique_ptr<lfm::BpeTokenizer> engine;
    mutable std::string error;
};

namespace {
thread_local std::string global_error;

lfm25_status fail(lfm25_model* model, lfm25_status status,
                  const std::string& message) noexcept {
    if (model) model->error = message;
    else global_error = message;
    return status;
}

template <typename Function>
lfm25_status protect(lfm25_model* model, Function&& function) noexcept {
    if (!model || !model->engine) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT,
                    "model handle is null");
    }
    try {
        function();
        model->error.clear();
        return LFM25_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(model, LFM25_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return fail(model, LFM25_STATUS_RUNTIME_ERROR, "unknown runtime error");
    }
}

lfm25_status tokenizer_fail(lfm25_tokenizer* tokenizer,
                              lfm25_status status,
                              const std::string& message) noexcept {
    if (tokenizer) tokenizer->error = message;
    else global_error = message;
    return status;
}

template <typename Function>
lfm25_status protect_tokenizer(lfm25_tokenizer* tokenizer,
                               Function&& function) noexcept {
    if (!tokenizer || !tokenizer->engine) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_INVALID_ARGUMENT,
                              "tokenizer handle is null");
    }
    try {
        function();
        tokenizer->error.clear();
        return LFM25_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_RUNTIME_ERROR,
                              "unknown tokenizer error");
    }
}

bool valid_struct(uint32_t actual, size_t expected) {
    return actual >= expected;
}

lfm::ModelOptions convert_model_options(const lfm25_model_options_v2& source) {
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

    // Expert offload options
    lfm::ExpertOffloadOptions& off = result.expert_offload;
    if (source.expert_offload_mode == 1) {
        off.mode = lfm::ExpertOffloadMode::Auto;
    } else if (source.expert_offload_mode == 2) {
        off.mode = lfm::ExpertOffloadMode::Host;
    } else {
        off.mode = lfm::ExpertOffloadMode::None;
    }

    if (source.expert_host_mode == 1) {
        off.host_mode = lfm::ExpertHostMode::PinnedCopy;
    } else if (source.expert_host_mode == 2) {
        off.host_mode = lfm::ExpertHostMode::Staged;
    } else {
        off.host_mode = lfm::ExpertHostMode::Mapped;
    }

    if (source.expert_cache_policy == 0) {
        off.policy = lfm::ExpertCachePolicy::Static;
    } else if (source.expert_cache_policy == 1) {
        off.policy = lfm::ExpertCachePolicy::Lru;
    } else if (source.expert_cache_policy == 3) {
        off.policy = lfm::ExpertCachePolicy::Score;
    } else {
        off.policy = lfm::ExpertCachePolicy::LayerLocalLfuLru;
    }

    off.gpu_expert_cache_bytes = source.gpu_expert_cache_bytes;
    off.experts_per_layer = source.experts_per_layer;
    off.maximum_pinned_host_bytes = source.maximum_pinned_host_bytes;
    off.gpu_memory_reserve_bytes = source.gpu_memory_reserve_bytes;
    off.prefill_chunk_tokens = source.prefill_chunk_tokens;
    off.prefetch_experts = source.prefetch_experts;

    if (source.expert_backing == 1) {
        off.backing = lfm::ExpertBackingMode::DiskCached;
    } else {
        off.backing = lfm::ExpertBackingMode::HostResident;
    }

    off.host_expert_cache_bytes = source.host_expert_cache_bytes;

    if (source.expert_io_backend == 1) {
        off.io_backend = lfm::ExpertIoBackend::ThreadPool;
    } else if (source.expert_io_backend == 2) {
        off.io_backend = lfm::ExpertIoBackend::IoUring;
    } else if (source.expert_io_backend == 3) {
        off.io_backend = lfm::ExpertIoBackend::WindowsOverlapped;
    } else {
        off.io_backend = lfm::ExpertIoBackend::Auto;
    }

    off.io_queue_depth = source.expert_io_queue_depth;
    off.io_workers = source.expert_io_workers;
    off.direct_io = source.expert_direct_io != 0;
    if (source.expert_sidecar_path) {
        off.expert_sidecar_path = source.expert_sidecar_path;
    }
    if (source.mirror_path) {
        off.mirror_path = source.mirror_path;
    }
    if (source.usage_profile_path) {
        off.usage_profile_path = source.usage_profile_path;
    }

    return result;
}

lfm::GenerationConfig convert_generation(
    const lfm25_generation_options_v1& source) {
    lfm::GenerationConfig result;
    result.temperature = source.temperature;
    result.top_k = source.top_k;
    result.top_p = source.top_p;
    result.repetition_penalty = source.repetition_penalty;
    result.seed = source.seed;
    return result;
}
} // namespace

extern "C" {

uint32_t lfm25_api_version(void) { return LFM25_C_API_VERSION; }

void lfm25_model_options_v2_init(lfm25_model_options_v2* options) {
    if (!options) return;
    std::memset(options, 0, sizeof(*options));
    options->struct_size = sizeof(*options);
    options->flags = LFM25_OPTION_FUSED_SAMPLING | LFM25_OPTION_CUDA_GRAPH;
    options->max_context = 4096;
    options->weight_mode = LFM25_WEIGHT_BF16;
    options->kv_cache_mode = LFM25_KV_BF16;
    options->gemm_backend = LFM25_GEMM_CUBLAS;
    options->attention_mode = LFM25_ATTENTION_SINGLE;
    options->attention_chunk_tokens = 256;
    options->attention_auto_threshold = 4096;
    options->lt_workspace_mb = 64;
    options->lt_heuristics = 8;

    options->expert_offload_mode = 0; // none
    options->expert_host_mode = 0;    // mapped
    options->expert_cache_policy = 2; // lfu-lru
    options->gpu_expert_cache_bytes = 0;
    options->experts_per_layer = 0;
    options->maximum_pinned_host_bytes = 9ULL * 1024 * 1024 * 1024;
    options->gpu_memory_reserve_bytes = 768ULL * 1024 * 1024;
    options->prefill_chunk_tokens = 256;
    options->prefetch_experts = 0;

    options->expert_backing = 0;      // host-resident
    options->host_expert_cache_bytes = 4ULL * 1024 * 1024 * 1024;
    options->expert_io_backend = 0;    // auto
    options->expert_io_queue_depth = 16;
    options->expert_io_workers = 4;
    options->expert_direct_io = 0;
    options->expert_sidecar_path = nullptr;
    options->mirror_path = nullptr;
    options->usage_profile_path = nullptr;
}

void lfm25_generation_options_init(lfm25_generation_options_v1* options) {
    if (!options) return;
    std::memset(options, 0, sizeof(*options));
    options->struct_size = sizeof(*options);
    options->temperature = 0.1f;
    options->top_k = 50;
    options->top_p = 1.0f;
    options->repetition_penalty = 1.05f;
    options->seed = 1;
}

lfm25_model* lfm25_model_create(
    const char* safetensors_path,
    const lfm25_model_options_v2* model_options,
    const lfm25_generation_options_v1* generation_options) {
    global_error.clear();
    if (!safetensors_path || !model_options || !generation_options) {
        global_error = "create requires non-null path and option structures";
        return nullptr;
    }
    if (!valid_struct(model_options->struct_size, sizeof(*model_options)) ||
        !valid_struct(generation_options->struct_size, sizeof(*generation_options))) {
        global_error = "option structure is smaller than API version 1";
        return nullptr;
    }
    try {
        auto handle = std::make_unique<lfm25_model>();
        handle->engine = std::make_unique<lfm::LfmModel>(
            safetensors_path, model_options->max_context,
            convert_model_options(*model_options),
            convert_generation(*generation_options));
        return handle.release();
    } catch (const std::exception& error) {
        global_error = error.what();
        return nullptr;
    } catch (...) {
        global_error = "unknown model creation error";
        return nullptr;
    }
}

void lfm25_model_destroy(lfm25_model* model) { delete model; }

const char* lfm25_last_error(const lfm25_model* model) {
    return model ? model->error.c_str() : global_error.c_str();
}

lfm25_status lfm25_reset(lfm25_model* model) {
    return protect(model, [&] { model->engine->session().reset(); });
}

lfm25_status lfm25_prefill(
    lfm25_model* model, const int32_t* tokens, size_t token_count) {
    if (!tokens || token_count == 0) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT,
                    "prefill requires at least one token");
    }
    return protect(model, [&] {
        model->engine->session().prefill(std::vector<int32_t>(tokens, tokens + token_count));
    });
}

lfm25_status lfm25_decode(lfm25_model* model, int32_t* token) {
    if (!token) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT,
                    "decode output pointer is null");
    }
    return protect(model, [&] { *token = model->engine->session().decode(); });
}

lfm25_status lfm25_position(const lfm25_model* model, int32_t* position) {
    if (!model || !model->engine || !position) {
        return fail(const_cast<lfm25_model*>(model),
                    LFM25_STATUS_INVALID_ARGUMENT,
                    "position requires a valid model and output pointer");
    }
    *position = model->engine->session().position();
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_model_vocab_size(const lfm25_model* model, int32_t* vocab_size) {
    if (!model || !model->engine || !vocab_size) {
        return fail(const_cast<lfm25_model*>(model),
                    LFM25_STATUS_INVALID_ARGUMENT,
                    "vocab_size requires a valid model and output pointer");
    }
    *vocab_size = model->engine->diagnostics().vocab_size();
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_copy_logits(
    lfm25_model* model, float* logits, size_t logits_count) {
    if (!model || !model->engine) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT, "model handle is null");
    }
    const int32_t vocab = model->engine->diagnostics().vocab_size();
    if (!logits || logits_count < static_cast<size_t>(vocab)) {
        return fail(model, LFM25_STATUS_BUFFER_TOO_SMALL,
                    "logits buffer must contain at least vocab_size floats");
    }
    return protect(model, [&] {
        const std::vector<float> values = model->engine->diagnostics().copy_logits();
        std::copy(values.begin(), values.end(), logits);
    });
}

lfm25_status lfm25_generate(
    lfm25_model* model, int32_t max_new_tokens, int32_t eos_token,
    lfm25_token_callback callback, void* user_data, int32_t* generated_tokens) {
    if (max_new_tokens < 0) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT,
                    "max_new_tokens must be non-negative");
    }
    if (generated_tokens) *generated_tokens = 0;
    if (!model || !model->engine) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT, "model handle is null");
    }
    try {
        int32_t count = 0;
        for (; count < max_new_tokens; ++count) {
            const int32_t token = model->engine->session().decode();
            if (token == eos_token) break;
            if (callback && callback(token, user_data) != 0) {
                if (generated_tokens) *generated_tokens = count + 1;
                model->error.clear();
                return LFM25_STATUS_STOPPED;
            }
        }
        if (generated_tokens) *generated_tokens = count;
        model->error.clear();
        return LFM25_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        return fail(model, LFM25_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(model, LFM25_STATUS_RUNTIME_ERROR, error.what());
    }
}

lfm25_status lfm25_save_session(lfm25_model* model, const char* path) {
    if (!path) return fail(model, LFM25_STATUS_INVALID_ARGUMENT, "session path is null");
    return protect(model, [&] { model->engine->persistence().save_session(path); });
}

lfm25_status lfm25_load_session(lfm25_model* model, const char* path) {
    if (!path) return fail(model, LFM25_STATUS_INVALID_ARGUMENT, "session path is null");
    return protect(model, [&] { model->engine->persistence().load_session(path); });
}

lfm25_status lfm25_get_memory_stats(
    const lfm25_model* model, lfm25_memory_stats_v1* stats) {
    if (!model || !model->engine || !stats ||
        !valid_struct(stats->struct_size, sizeof(*stats))) {
        return fail(const_cast<lfm25_model*>(model), LFM25_STATUS_INVALID_ARGUMENT,
                    "invalid memory stats request");
    }
    const lfm::ModelMemoryStats value = model->engine->diagnostics().memory_stats();
    stats->weights = value.weights;
    stats->kv_cache = value.kv_cache;
    stats->conv_state = value.conv_state;
    stats->rope_tables = value.rope_tables;
    stats->activations = value.activations;
    stats->sampling = value.sampling;
    stats->matmul_workspace = value.matmul_workspace;
    stats->attention_workspace = value.attention_workspace;
    stats->total = value.total();
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_get_runtime_metrics(
    const lfm25_model* model, lfm25_runtime_metrics_v1* metrics) {
    if (!model || !model->engine || !metrics ||
        !valid_struct(metrics->struct_size, sizeof(*metrics))) {
        return fail(const_cast<lfm25_model*>(model), LFM25_STATUS_INVALID_ARGUMENT,
                    "invalid runtime metrics request");
    }
    const lfm::RuntimeMetrics value = model->engine->diagnostics().runtime_metrics();
    metrics->last_prefill_ms = value.last_prefill_ms;
    metrics->prefill_tokens = value.prefill_tokens;
    metrics->prefill_tokens_per_second = value.prefill_tokens_per_second();
    metrics->cumulative_decode_ms = value.cumulative_decode_ms;
    metrics->decoded_tokens = value.decoded_tokens;
    metrics->decode_tokens_per_second = value.decode_tokens_per_second();
    return LFM25_STATUS_OK;
}

lfm25_tokenizer* lfm25_tokenizer_create(const char* tokenizer_json_path) {
    global_error.clear();
    if (!tokenizer_json_path) {
        global_error = "tokenizer path is null";
        return nullptr;
    }
    try {
        auto handle = std::make_unique<lfm25_tokenizer>();
        handle->engine =
            std::make_unique<lfm::BpeTokenizer>(tokenizer_json_path);
        return handle.release();
    } catch (const std::exception& error) {
        global_error = error.what();
        return nullptr;
    } catch (...) {
        global_error = "unknown tokenizer creation error";
        return nullptr;
    }
}

void lfm25_tokenizer_destroy(lfm25_tokenizer* tokenizer) { delete tokenizer; }

const char* lfm25_tokenizer_last_error(const lfm25_tokenizer* tokenizer) {
    return tokenizer ? tokenizer->error.c_str() : global_error.c_str();
}

int32_t lfm25_tokenizer_bos_id(const lfm25_tokenizer* tokenizer) {
    return tokenizer && tokenizer->engine ? tokenizer->engine->bos_id() : -1;
}

int32_t lfm25_tokenizer_eos_id(const lfm25_tokenizer* tokenizer) {
    return tokenizer && tokenizer->engine ? tokenizer->engine->eos_id() : -1;
}

lfm25_status lfm25_tokenizer_encode(
    lfm25_tokenizer* tokenizer, const char* text, int add_bos,
    int32_t* token_ids, size_t token_capacity, size_t* required_tokens) {
    if (!text || !required_tokens) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_INVALID_ARGUMENT,
                              "encode requires text and required_tokens");
    }
    std::vector<int32_t> values;
    const lfm25_status status = protect_tokenizer(tokenizer, [&] {
        values = tokenizer->engine->encode(text, add_bos != 0);
    });
    if (status != LFM25_STATUS_OK) return status;
    *required_tokens = values.size();
    if (!token_ids || token_capacity < values.size()) {
        tokenizer->error = "token buffer is too small";
        return LFM25_STATUS_BUFFER_TOO_SMALL;
    }
    std::copy(values.begin(), values.end(), token_ids);
    tokenizer->error.clear();
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_tokenizer_decode(
    lfm25_tokenizer* tokenizer, const int32_t* token_ids, size_t token_count,
    int skip_special, char* text, size_t text_capacity, size_t* required_bytes) {
    if ((!token_ids && token_count != 0) || !required_bytes) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_INVALID_ARGUMENT,
                              "decode requires token data and required_bytes");
    }
    std::string value;
    const lfm25_status status = protect_tokenizer(tokenizer, [&] {
        std::vector<int32_t> ids;
        if (token_count != 0) ids.assign(token_ids, token_ids + token_count);
        value = tokenizer->engine->decode(ids, skip_special != 0);
    });
    if (status != LFM25_STATUS_OK) return status;
    *required_bytes = value.size() + 1;
    if (!text || text_capacity < value.size() + 1) {
        tokenizer->error = "text buffer is too small";
        return LFM25_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(text, value.c_str(), value.size() + 1);
    tokenizer->error.clear();
    return LFM25_STATUS_OK;
}

lfm25_status lfm25_tokenizer_format_chat(
    lfm25_tokenizer* tokenizer, const char* user_prompt,
    const char* system_prompt, char* text, size_t text_capacity,
    size_t* required_bytes) {
    if (!user_prompt || !required_bytes) {
        return tokenizer_fail(tokenizer, LFM25_STATUS_INVALID_ARGUMENT,
                              "format_chat requires a user prompt");
    }
    std::string value;
    const lfm25_status status = protect_tokenizer(tokenizer, [&] {
        value = tokenizer->engine->format_chat(
            user_prompt, system_prompt ? system_prompt : "");
    });
    if (status != LFM25_STATUS_OK) return status;
    *required_bytes = value.size() + 1;
    if (!text || text_capacity < value.size() + 1) {
        tokenizer->error = "text buffer is too small";
        return LFM25_STATUS_BUFFER_TOO_SMALL;
    }
    std::memcpy(text, value.c_str(), value.size() + 1);
    tokenizer->error.clear();
    return LFM25_STATUS_OK;
}

} // extern "C"
