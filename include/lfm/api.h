#ifndef LFM25_API_H
#define LFM25_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LFM25_API_BUILD)
#define LFM25_API __declspec(dllexport)
#elif defined(_WIN32)
#define LFM25_API __declspec(dllimport)
#else
#define LFM25_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum lfm25_backend { LFM25_BACKEND_CPU = 0, LFM25_BACKEND_CUDA = 1 } lfm25_backend;
typedef enum lfm25_status {
    LFM25_STATUS_OK = 0,
    LFM25_STATUS_INVALID_ARGUMENT = 1,
    LFM25_STATUS_RUNTIME_ERROR = 2,
    LFM25_STATUS_BUFFER_TOO_SMALL = 3,
    LFM25_STATUS_NOT_FOUND = 4,
    LFM25_STATUS_BACKEND_UNAVAILABLE = 5
} lfm25_status;
typedef enum lfm25_request_status {
    LFM25_REQUEST_QUEUED = 0, LFM25_REQUEST_PREFILLING = 1,
    LFM25_REQUEST_DECODING = 2, LFM25_REQUEST_COMPLETED = 3,
    LFM25_REQUEST_CANCELLED = 4, LFM25_REQUEST_FAILED = 5
} lfm25_request_status;

typedef struct lfm25_model lfm25_model;
typedef struct lfm25_engine lfm25_engine;
typedef struct lfm25_tokenizer lfm25_tokenizer;
typedef uint64_t lfm25_request_id;

typedef struct lfm25_generation_options {
    uint32_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} lfm25_generation_options;

typedef struct lfm25_cpu_model_config {
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
} lfm25_cpu_model_config;

typedef struct lfm25_cuda_model_options {
    uint32_t flags;
    int32_t weight_mode;
    int32_t kv_cache_mode;
    int32_t gemm_backend;
    int32_t attention_mode;
    int32_t attention_chunk_tokens;
    int32_t attention_auto_threshold;
    int32_t lt_workspace_mb;
    int32_t lt_heuristics;
} lfm25_cuda_model_options;

typedef struct lfm25_engine_model_options {
    uint32_t struct_size;
    lfm25_backend backend;
    int32_t max_context;
    union { lfm25_cpu_model_config cpu; lfm25_cuda_model_options cuda; } backend_options;
    lfm25_generation_options generation;
} lfm25_engine_model_options;

typedef struct lfm25_cpu_model_options {
    uint32_t struct_size;
    int32_t max_context;
    lfm25_cpu_model_config cpu;
    lfm25_generation_options generation;
} lfm25_cpu_model_options;

typedef struct lfm25_cpu_engine_options {
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
} lfm25_cpu_engine_options;

typedef struct lfm25_cuda_engine_options {
    int32_t max_active_requests;
    int32_t max_batched_tokens;
    int32_t prefill_chunk_tokens;
    int32_t page_tokens;
    uint64_t logical_kv_pages;
    int32_t scheduler_policy;
} lfm25_cuda_engine_options;

typedef struct lfm25_engine_options {
    uint32_t struct_size;
    lfm25_backend backend;
    union { lfm25_cpu_engine_options cpu; lfm25_cuda_engine_options cuda; } backend_options;
    lfm25_engine_model_options model;
} lfm25_engine_options;

typedef struct lfm25_request_options {
    uint32_t struct_size;
    uint32_t max_new_tokens;
    int32_t eos_token_id;
    int32_t priority;
    lfm25_generation_options generation;
} lfm25_request_options;

typedef struct lfm25_runtime_metrics {
    uint32_t struct_size;
    double prefill_ms;
    uint64_t prefill_tokens;
    double decode_ms;
    uint64_t decode_tokens;
} lfm25_runtime_metrics;

LFM25_API void lfm25_cpu_model_options_init(lfm25_cpu_model_options* options);
LFM25_API void lfm25_engine_options_init(lfm25_engine_options* options, lfm25_backend backend);
LFM25_API void lfm25_request_options_init(lfm25_request_options* options);
LFM25_API const char* lfm25_backend_capabilities(lfm25_backend backend);

LFM25_API lfm25_model* lfm25_model_create(const char* path, const lfm25_cpu_model_options* options);
LFM25_API void lfm25_model_destroy(lfm25_model* model);
LFM25_API lfm25_status lfm25_model_prefill(lfm25_model* model, const int32_t* tokens, size_t count);
LFM25_API lfm25_status lfm25_model_decode(lfm25_model* model, int32_t* token);
LFM25_API lfm25_status lfm25_model_copy_logits(lfm25_model* model, float* output, size_t capacity, size_t* required);
LFM25_API lfm25_status lfm25_model_get_metrics(lfm25_model* model, lfm25_runtime_metrics* metrics);
LFM25_API const char* lfm25_model_last_error(const lfm25_model* model);

LFM25_API lfm25_engine* lfm25_engine_create(const char* path, const lfm25_engine_options* options);
LFM25_API void lfm25_engine_destroy(lfm25_engine* engine);
LFM25_API lfm25_status lfm25_engine_submit(lfm25_engine* engine, const int32_t* tokens, size_t count, const lfm25_request_options* options, lfm25_request_id* request_id);
LFM25_API lfm25_status lfm25_engine_poll(lfm25_engine* engine, lfm25_request_id request_id, int32_t* output, size_t capacity, size_t* count, int* finished);
LFM25_API lfm25_status lfm25_engine_status(lfm25_engine* engine, lfm25_request_id request_id, lfm25_request_status* status);
LFM25_API lfm25_status lfm25_engine_cancel(lfm25_engine* engine, lfm25_request_id request_id);
LFM25_API lfm25_status lfm25_engine_step(lfm25_engine* engine, int* progressed);
LFM25_API const char* lfm25_engine_last_error(const lfm25_engine* engine);

LFM25_API lfm25_tokenizer* lfm25_tokenizer_create(const char* path);
LFM25_API void lfm25_tokenizer_destroy(lfm25_tokenizer* tokenizer);
LFM25_API lfm25_status lfm25_tokenizer_encode(lfm25_tokenizer* tokenizer, const char* text, int add_bos, int32_t* output, size_t capacity, size_t* required);
LFM25_API lfm25_status lfm25_tokenizer_decode(lfm25_tokenizer* tokenizer, const int32_t* tokens, size_t count, int skip_special, char* output, size_t capacity, size_t* required);
LFM25_API const char* lfm25_tokenizer_last_error(const lfm25_tokenizer* tokenizer);

#ifdef __cplusplus
}
#endif

#endif
