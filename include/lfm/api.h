#ifndef RUNTIME_API_H
#define RUNTIME_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(RUNTIME_API_BUILD)
#define RUNTIME_API __declspec(dllexport)
#elif defined(_WIN32)
#define RUNTIME_API __declspec(dllimport)
#else
#define RUNTIME_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum runtime_backend { RUNTIME_BACKEND_CPU = 0, RUNTIME_BACKEND_CUDA = 1 } runtime_backend;
typedef enum runtime_status {
    RUNTIME_STATUS_OK = 0,
    RUNTIME_STATUS_INVALID_ARGUMENT = 1,
    RUNTIME_STATUS_RUNTIME_ERROR = 2,
    RUNTIME_STATUS_BUFFER_TOO_SMALL = 3,
    RUNTIME_STATUS_NOT_FOUND = 4,
    RUNTIME_STATUS_BACKEND_UNAVAILABLE = 5
} runtime_status;
typedef enum runtime_request_status {
    RUNTIME_REQUEST_QUEUED = 0, RUNTIME_REQUEST_PREFILLING = 1,
    RUNTIME_REQUEST_DECODING = 2, RUNTIME_REQUEST_COMPLETED = 3,
    RUNTIME_REQUEST_CANCELLED = 4, RUNTIME_REQUEST_FAILED = 5
} runtime_request_status;

typedef struct runtime_model runtime_model;
typedef struct runtime_engine runtime_engine;
typedef struct runtime_tokenizer runtime_tokenizer;
typedef uint64_t runtime_request_id;

typedef struct runtime_generation_options {
    uint32_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} runtime_generation_options;

typedef struct runtime_cpu_model_config {
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
} runtime_cpu_model_config;

typedef struct runtime_cuda_model_options {
    uint32_t flags;
    int32_t weight_mode;
    int32_t kv_cache_mode;
    int32_t gemm_backend;
    int32_t attention_mode;
    int32_t attention_chunk_tokens;
    int32_t attention_auto_threshold;
    int32_t lt_workspace_mb;
    int32_t lt_heuristics;
} runtime_cuda_model_options;

typedef struct runtime_engine_model_options {
    uint32_t struct_size;
    runtime_backend backend;
    int32_t max_context;
    union { runtime_cpu_model_config cpu; runtime_cuda_model_options cuda; } backend_options;
    runtime_generation_options generation;
} runtime_engine_model_options;

typedef struct runtime_cpu_model_options {
    uint32_t struct_size;
    int32_t max_context;
    runtime_cpu_model_config cpu;
    runtime_generation_options generation;
} runtime_cpu_model_options;

typedef struct runtime_cpu_engine_options {
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
} runtime_cpu_engine_options;

typedef struct runtime_cuda_engine_options {
    int32_t max_active_requests;
    int32_t max_batched_tokens;
    int32_t prefill_chunk_tokens;
    int32_t page_tokens;
    uint64_t logical_kv_pages;
    int32_t scheduler_policy;
} runtime_cuda_engine_options;

typedef struct runtime_engine_options {
    uint32_t struct_size;
    runtime_backend backend;
    union { runtime_cpu_engine_options cpu; runtime_cuda_engine_options cuda; } backend_options;
    runtime_engine_model_options model;
} runtime_engine_options;

typedef struct runtime_request_options {
    uint32_t struct_size;
    uint32_t max_new_tokens;
    int32_t eos_token_id;
    int32_t priority;
    runtime_generation_options generation;
} runtime_request_options;

typedef struct runtime_metrics {
    uint32_t struct_size;
    double prefill_ms;
    uint64_t prefill_tokens;
    double decode_ms;
    uint64_t decode_tokens;
} runtime_metrics;

RUNTIME_API void runtime_cpu_model_options_init(runtime_cpu_model_options* options);
RUNTIME_API void runtime_engine_options_init(runtime_engine_options* options, runtime_backend backend);
RUNTIME_API void runtime_request_options_init(runtime_request_options* options);
RUNTIME_API const char* runtime_backend_capabilities(runtime_backend backend);

RUNTIME_API runtime_model* runtime_model_create(const char* path, const runtime_cpu_model_options* options);
RUNTIME_API void runtime_model_destroy(runtime_model* model);
RUNTIME_API runtime_status runtime_model_prefill(runtime_model* model, const int32_t* tokens, size_t count);
RUNTIME_API runtime_status runtime_model_decode(runtime_model* model, int32_t* token);
RUNTIME_API runtime_status runtime_model_copy_logits(runtime_model* model, float* output, size_t capacity, size_t* required);
RUNTIME_API runtime_status runtime_model_get_metrics(runtime_model* model, runtime_metrics* metrics);
RUNTIME_API const char* runtime_model_last_error(const runtime_model* model);

RUNTIME_API runtime_engine* runtime_engine_create(const char* path, const runtime_engine_options* options);
RUNTIME_API void runtime_engine_destroy(runtime_engine* engine);
RUNTIME_API runtime_status runtime_engine_submit(runtime_engine* engine, const int32_t* tokens, size_t count, const runtime_request_options* options, runtime_request_id* request_id);
RUNTIME_API runtime_status runtime_engine_poll(runtime_engine* engine, runtime_request_id request_id, int32_t* output, size_t capacity, size_t* count, int* finished);
RUNTIME_API runtime_status runtime_engine_status(runtime_engine* engine, runtime_request_id request_id, runtime_request_status* status);
RUNTIME_API runtime_status runtime_engine_cancel(runtime_engine* engine, runtime_request_id request_id);
RUNTIME_API runtime_status runtime_engine_step(runtime_engine* engine, int* progressed);
RUNTIME_API const char* runtime_engine_last_error(const runtime_engine* engine);

RUNTIME_API runtime_tokenizer* runtime_tokenizer_create(const char* path);
RUNTIME_API void runtime_tokenizer_destroy(runtime_tokenizer* tokenizer);
RUNTIME_API runtime_status runtime_tokenizer_encode(runtime_tokenizer* tokenizer, const char* text, int add_bos, int32_t* output, size_t capacity, size_t* required);
RUNTIME_API runtime_status runtime_tokenizer_decode(runtime_tokenizer* tokenizer, const int32_t* tokens, size_t count, int skip_special, char* output, size_t capacity, size_t* required);
RUNTIME_API const char* runtime_tokenizer_last_error(const runtime_tokenizer* tokenizer);

#ifdef __cplusplus
}
#endif

#endif
