#include "api_internal.hpp"

extern "C" {

void celeg_cpu_model_options_init(celeg_cpu_model_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_context = 4096;
    options->generation.struct_size = sizeof(options->generation);
    options->generation.temperature = 0.1f;
    options->generation.top_k = 50;
    options->generation.top_p = 1.0f;
    options->generation.repetition_penalty = 1.05f;
    options->generation.seed = 1;
    options->cpu.q4_group_size = 32;
    options->cpu.use_pack_cache = 1;
    options->cpu.kv_cache_mode = 1;
    options->cpu.kv_page_tokens = 32;
    options->cpu.prefill_chunk_tokens = 256;
    options->cpu.prefill_chunk_threshold = 16;
    options->cpu.attention_parallel_threshold = 256;
    options->cpu.attention_page_tile = 4;
}

void celeg_engine_options_init(celeg_engine_options* options,
                               const char* backend_id,
                               const void* backend_options,
                               uint32_t backend_options_size) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->backend_id = backend_id;
    options->max_context = 4096;
    options->backend_options = backend_options;
    options->backend_options_size = backend_options_size;
    options->generation.struct_size = sizeof(options->generation);
    options->generation.temperature = 0.1f;
    options->generation.top_k = 50;
    options->generation.top_p = 1.0f;
    options->generation.repetition_penalty = 1.05f;
    options->generation.seed = 1;
}

void celeg_cpu_backend_options_init(celeg_cpu_backend_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->model.q4_group_size = 32;
    options->model.use_pack_cache = 1;
    options->model.kv_cache_mode = 1;
    options->model.kv_page_tokens = 32;
    options->model.prefill_chunk_tokens = 256;
    options->model.prefill_chunk_threshold = 16;
    options->model.attention_parallel_threshold = 256;
    options->model.attention_page_tile = 4;
    options->engine.max_active_requests = 16;
    options->engine.max_batched_tokens = 256;
    options->engine.max_prefill_batch = 16;
    options->engine.max_decode_batch = 16;
    options->engine.decode_first = 1;
    options->engine.long_prefill_chunk_tokens = 256;
    options->engine.long_prefill_threshold = 32;
    options->engine.prefix_cache = 1;
    options->engine.prefix_cache_max_entries = 256;
    options->engine.prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
}

void celeg_cuda_backend_options_init(celeg_cuda_backend_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->model.weight_mode = 0;
    options->model.kv_cache_mode = 0;
    options->model.gemm_backend = 0;
    options->model.attention_mode = 2;
    options->model.attention_chunk_tokens = 32;
    options->model.attention_auto_threshold = 1;
    options->model.lt_workspace_mb = 64;
    options->model.lt_heuristics = 8;
    options->engine.max_active_requests = 8;
    options->engine.max_batched_tokens = 512;
    options->engine.prefill_chunk_tokens = 256;
    options->engine.page_tokens = 16;
}

void celeg_request_options_init(celeg_request_options* options) {
    if (!options) return;
    *options = {};
    options->struct_size = sizeof(*options);
    options->max_new_tokens = 128;
    options->eos_token_id = 7;
    options->generation.struct_size = sizeof(options->generation);
    options->generation.temperature = 0.1f;
    options->generation.top_k = 50;
    options->generation.top_p = 1.0f;
    options->generation.repetition_penalty = 1.05f;
    options->generation.seed = 1;
}

const char* celeg_backend_capabilities(celeg_backend backend) {
    if (backend == CELEG_BACKEND_CPU) {
        celeg::api::global_error = celeg::detect_cpu_capabilities().summary();
        return celeg::api::global_error.c_str();
    }
#ifdef CELEG_API_WITH_CUDA
    celeg::api::global_error =
        "CUDA backend available for celeg_engine_*; celeg_model_* remains CPU-only";
    return celeg::api::global_error.c_str();
#else
    return "CUDA backend unavailable in this build";
#endif
}

} // extern "C"
