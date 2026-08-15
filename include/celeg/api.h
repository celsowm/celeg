#ifndef CELEG_API_H
#define CELEG_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CELEG_API_BUILD)
#define CELEG_API __declspec(dllexport)
#elif defined(_WIN32)
#define CELEG_API __declspec(dllimport)
#else
#define CELEG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum celeg_backend { CELEG_BACKEND_CPU = 0, CELEG_BACKEND_CUDA = 1 } celeg_backend;
typedef enum celeg_status {
    CELEG_STATUS_OK = 0,
    CELEG_STATUS_INVALID_ARGUMENT = 1,
    CELEG_STATUS_CELEG_ERROR = 2,
    CELEG_STATUS_BUFFER_TOO_SMALL = 3,
    CELEG_STATUS_NOT_FOUND = 4,
    CELEG_STATUS_BACKEND_UNAVAILABLE = 5
} celeg_status;
typedef enum celeg_request_status {
    CELEG_REQUEST_QUEUED = 0, CELEG_REQUEST_PREFILLING = 1,
    CELEG_REQUEST_DECODING = 2, CELEG_REQUEST_COMPLETED = 3,
    CELEG_REQUEST_CANCELLED = 4, CELEG_REQUEST_FAILED = 5
} celeg_request_status;

typedef struct celeg_model celeg_model;
typedef struct celeg_engine celeg_engine;
typedef struct celeg_tokenizer celeg_tokenizer;
typedef uint64_t celeg_request_id;

typedef struct celeg_generation_options {
    uint32_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} celeg_generation_options;

typedef struct celeg_cpu_model_config {
    int32_t threads;
    int32_t isa;
    int32_t q4_group_size;
    int32_t use_pack_cache;
    const char* pack_cache_directory;
    int32_t affinity;
    int32_t kv_cache_mode;
    uint32_t kv_page_tokens;
    uint32_t prefill_chunk_tokens;
    uint32_t prefill_chunk_threshold;
    uint32_t attention_parallel_threshold;
    uint32_t attention_page_tile;
    int32_t numa_mode;
} celeg_cpu_model_config;

typedef struct celeg_cuda_model_options {
    uint32_t flags;
    int32_t weight_mode;
    int32_t kv_cache_mode;
    int32_t gemm_backend;
    int32_t attention_mode;
    int32_t attention_chunk_tokens;
    int32_t attention_auto_threshold;
    int32_t lt_workspace_mb;
    int32_t lt_heuristics;
} celeg_cuda_model_options;

typedef struct celeg_cpu_model_options {
    uint32_t struct_size;
    int32_t max_context;
    celeg_cpu_model_config cpu;
    celeg_generation_options generation;
} celeg_cpu_model_options;

typedef struct celeg_cpu_engine_options {
    uint32_t max_active_requests;
    uint32_t max_batched_tokens;
    uint32_t max_prefill_batch;
    uint32_t max_decode_batch;
    int32_t decode_first;
    uint32_t long_prefill_chunk_tokens;
    uint32_t long_prefill_threshold;
    int32_t prefix_cache;
    uint32_t prefix_cache_max_entries;
    uint64_t prefix_cache_max_bytes;
} celeg_cpu_engine_options;

typedef struct celeg_cuda_engine_options {
    int32_t max_active_requests;
    int32_t max_batched_tokens;
    int32_t prefill_chunk_tokens;
    int32_t page_tokens;
    uint64_t logical_kv_pages;
    int32_t scheduler_policy;
} celeg_cuda_engine_options;

typedef struct celeg_engine_options {
    uint32_t struct_size;
    const char* backend_id;
    int32_t max_context;
    const void* backend_options;
    uint32_t backend_options_size;
    celeg_generation_options generation;
} celeg_engine_options;

typedef struct celeg_cpu_backend_options {
    uint32_t struct_size;
    celeg_cpu_model_config model;
    celeg_cpu_engine_options engine;
} celeg_cpu_backend_options;

typedef struct celeg_cuda_backend_options {
    uint32_t struct_size;
    celeg_cuda_model_options model;
    celeg_cuda_engine_options engine;
} celeg_cuda_backend_options;

typedef struct celeg_request_options {
    uint32_t struct_size;
    uint32_t max_new_tokens;
    int32_t eos_token_id;
    int32_t priority;
    celeg_generation_options generation;
} celeg_request_options;

typedef struct celeg_metrics {
    uint32_t struct_size;
    double prefill_ms;
    uint64_t prefill_tokens;
    double decode_ms;
    uint64_t decode_tokens;
} celeg_metrics;

CELEG_API void celeg_cpu_model_options_init(celeg_cpu_model_options* options);
CELEG_API void celeg_engine_options_init(celeg_engine_options* options,
                                          const char* backend_id,
                                          const void* backend_options,
                                          uint32_t backend_options_size);
CELEG_API void celeg_cpu_backend_options_init(celeg_cpu_backend_options* options);
CELEG_API void celeg_cuda_backend_options_init(celeg_cuda_backend_options* options);
CELEG_API void celeg_request_options_init(celeg_request_options* options);
CELEG_API const char* celeg_backend_capabilities(celeg_backend backend);

CELEG_API celeg_model* celeg_model_create(const char* path, const celeg_cpu_model_options* options);
CELEG_API void celeg_model_destroy(celeg_model* model);
CELEG_API celeg_status celeg_model_prefill(celeg_model* model, const int32_t* tokens, size_t count);
CELEG_API celeg_status celeg_model_decode(celeg_model* model, int32_t* token);
CELEG_API celeg_status celeg_model_copy_logits(celeg_model* model, float* output, size_t capacity, size_t* required);
CELEG_API celeg_status celeg_model_get_metrics(celeg_model* model, celeg_metrics* metrics);
CELEG_API const char* celeg_model_last_error(const celeg_model* model);

CELEG_API celeg_engine* celeg_engine_create(const char* path, const celeg_engine_options* options);
CELEG_API void celeg_engine_destroy(celeg_engine* engine);
CELEG_API celeg_status celeg_engine_submit(celeg_engine* engine, const int32_t* tokens, size_t count, const celeg_request_options* options, celeg_request_id* request_id);
CELEG_API celeg_status celeg_engine_poll(celeg_engine* engine, celeg_request_id request_id, int32_t* output, size_t capacity, size_t* count, int* finished);
CELEG_API celeg_status celeg_engine_status(celeg_engine* engine, celeg_request_id request_id, celeg_request_status* status);
CELEG_API celeg_status celeg_engine_cancel(celeg_engine* engine, celeg_request_id request_id);
CELEG_API celeg_status celeg_engine_step(celeg_engine* engine, int* progressed);
CELEG_API const char* celeg_engine_last_error(const celeg_engine* engine);

CELEG_API celeg_tokenizer* celeg_tokenizer_create(const char* path);
CELEG_API void celeg_tokenizer_destroy(celeg_tokenizer* tokenizer);
CELEG_API celeg_status celeg_tokenizer_encode(celeg_tokenizer* tokenizer, const char* text, int add_bos, int32_t* output, size_t capacity, size_t* required);
CELEG_API celeg_status celeg_tokenizer_decode(celeg_tokenizer* tokenizer, const int32_t* tokens, size_t count, int skip_special, char* output, size_t capacity, size_t* required);
CELEG_API const char* celeg_tokenizer_last_error(const celeg_tokenizer* tokenizer);

#ifdef __cplusplus
}
#endif

#endif
