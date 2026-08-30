#include "options_defaults.hpp"

namespace celeg::api::defaults {

void generation(celeg_generation_options& options) {
    options.struct_size = sizeof(options);
    options.temperature = 0.1f;
    options.top_k = 50;
    options.top_p = 1.0f;
    options.repetition_penalty = 1.05f;
    options.seed = 1;
}

void cpu_model_config(celeg_cpu_model_config& options) {
    options.q4_group_size = 32;
    options.use_pack_cache = 1;
    options.kv_cache_mode = CELEG_CPU_KV_CACHE_BF16;
    options.kv_page_tokens = 32;
    options.prefill_chunk_tokens = 256;
    options.prefill_chunk_threshold = 16;
    options.attention_parallel_threshold = 256;
    options.attention_page_tile = 4;
}

void cpu_engine_options(celeg_cpu_engine_options& options) {
    options.max_active_requests = 16;
    options.max_batched_tokens = 256;
    options.max_prefill_batch = 16;
    options.max_decode_batch = 16;
    options.decode_first = 1;
    options.long_prefill_chunk_tokens = 256;
    options.long_prefill_threshold = 32;
    options.prefix_cache = 1;
    options.prefix_cache_max_entries = 256;
    options.prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
}

void cuda_model_options(celeg_cuda_model_options& options) {
    options.weight_mode = CELEG_WEIGHT_MODE_BF16;
    options.kv_cache_mode = CELEG_CUDA_KV_CACHE_BF16;
    options.gemm_backend = CELEG_CUDA_GEMM_CUBLAS;
    options.attention_mode = CELEG_CUDA_ATTENTION_AUTO;
    options.attention_chunk_tokens = 32;
    options.attention_auto_threshold = 1;
    options.lt_workspace_mb = 64;
    options.lt_heuristics = 8;
}

void cuda_engine_options(celeg_cuda_engine_options& options) {
    options.max_active_requests = 8;
    options.max_batched_tokens = 512;
    options.prefill_chunk_tokens = 256;
    options.page_tokens = 16;
}

void metal_model_options(celeg_metal_model_options& options) {
    options.weight_mode = CELEG_METAL_WEIGHT_BF16;
    options.kv_cache_mode = CELEG_METAL_KV_CACHE_BF16;
    options.storage_mode = CELEG_METAL_STORAGE_SHARED;
    options.kv_page_tokens = 16;
}

void metal_engine_options(celeg_metal_engine_options& options) {
    options.max_active_requests = 1;
    options.max_batched_tokens = 256;
    options.prefill_chunk_tokens = 256;
    options.kv_page_tokens = 16;
    options.prefix_cache = 1;
    options.prefix_cache_max_entries = 32;
}

}
