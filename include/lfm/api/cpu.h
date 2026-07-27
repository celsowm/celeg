#ifndef LFM25_CPU_C_API_H
#define LFM25_CPU_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LFM25_CPU_C_API_BUILD)
#define LFM25_CPU_API __declspec(dllexport)
#elif defined(_WIN32)
#define LFM25_CPU_API __declspec(dllimport)
#else
#define LFM25_CPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LFM25_CPU_C_API_VERSION 6u

typedef struct lfm25_cpu_model lfm25_cpu_model;
typedef struct lfm25_cpu_tokenizer lfm25_cpu_tokenizer;
typedef struct lfm25_cpu_engine lfm25_cpu_engine;
typedef uint64_t lfm25_cpu_request_id;

typedef enum lfm25_cpu_status {
    LFM25_CPU_STATUS_OK = 0,
    LFM25_CPU_STATUS_INVALID_ARGUMENT = 1,
    LFM25_CPU_STATUS_RUNTIME_ERROR = 2,
    LFM25_CPU_STATUS_BUFFER_TOO_SMALL = 3,
    LFM25_CPU_STATUS_NOT_FOUND = 4
} lfm25_cpu_status;

typedef enum lfm25_cpu_isa {
    LFM25_CPU_ISA_AUTO = 0,
    LFM25_CPU_ISA_SCALAR = 1,
    LFM25_CPU_ISA_AVX2 = 2,
    LFM25_CPU_ISA_AVX_VNNI = 3,
    LFM25_CPU_ISA_AVX512_VNNI = 4,
    LFM25_CPU_ISA_AMX_INT8 = 5,
    LFM25_CPU_ISA_NEON = 6,
    LFM25_CPU_ISA_DOTPROD = 7,
    LFM25_CPU_ISA_I8MM = 8,
    LFM25_CPU_ISA_SVE2 = 9,
    LFM25_CPU_ISA_SME2 = 10
} lfm25_cpu_isa;

typedef enum lfm25_cpu_affinity {
    LFM25_CPU_AFFINITY_NONE = 0,
    LFM25_CPU_AFFINITY_COMPACT = 1,
    LFM25_CPU_AFFINITY_SCATTER = 2
} lfm25_cpu_affinity;

typedef enum lfm25_cpu_kv_cache_mode {
    LFM25_CPU_KV_FP32 = 0,
    LFM25_CPU_KV_BF16 = 1
} lfm25_cpu_kv_cache_mode;

typedef enum lfm25_cpu_numa_mode {
    LFM25_CPU_NUMA_DISABLED = 0,
    LFM25_CPU_NUMA_LOCAL = 1,
    LFM25_CPU_NUMA_REPLICATE_WEIGHTS = 2
} lfm25_cpu_numa_mode;

typedef enum lfm25_cpu_request_status {
    LFM25_CPU_REQUEST_QUEUED = 0,
    LFM25_CPU_REQUEST_PREFILLING = 1,
    LFM25_CPU_REQUEST_DECODING = 2,
    LFM25_CPU_REQUEST_COMPLETED = 3,
    LFM25_CPU_REQUEST_CANCELLED = 4,
    LFM25_CPU_REQUEST_FAILED = 5
} lfm25_cpu_request_status;

// Single model options structure for the v6 C API. Earlier v1-v4 entry points
// were removed because the runtime no longer ships a stable ABI surface.
typedef struct lfm25_cpu_model_options_v6 {
    uint32_t struct_size;
    int32_t max_context;
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
} lfm25_cpu_model_options_v6;

typedef struct lfm25_cpu_generation_options_v1 {
    uint32_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} lfm25_cpu_generation_options_v1;

typedef struct lfm25_cpu_runtime_metrics_v1 {
    uint32_t struct_size;
    double prefill_ms;
    uint64_t prefill_tokens;
    double prefill_tokens_per_second;
    double decode_ms;
    uint64_t decode_tokens;
    double decode_tokens_per_second;
} lfm25_cpu_runtime_metrics_v1;

typedef struct lfm25_cpu_memory_stats_v2 {
    uint32_t struct_size;
    uint64_t weights;
    uint64_t kv_cache;
    uint64_t conv_state;
    uint64_t activations;
    uint64_t total;
    uint64_t kv_pages_used;
    uint64_t kv_pages_total;
} lfm25_cpu_memory_stats_v2;

typedef struct lfm25_cpu_engine_options_v3 {
    uint32_t struct_size;
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
} lfm25_cpu_engine_options_v3;

typedef struct lfm25_cpu_request_options_v1 {
    uint32_t struct_size;
    uint32_t max_new_tokens;
    int32_t eos_token_id;
    int32_t priority;
    lfm25_cpu_generation_options_v1 generation;
} lfm25_cpu_request_options_v1;

typedef struct lfm25_cpu_engine_metrics_v3 {
    uint32_t struct_size;
    uint64_t submitted_requests;
    uint64_t completed_requests;
    uint64_t cancelled_requests;
    uint64_t failed_requests;
    uint64_t active_requests;
    uint64_t queued_requests;
    uint64_t prefill_tokens;
    uint64_t decode_tokens;
    uint64_t ragged_prefill_steps;
    uint64_t packed_decode_steps;
    uint64_t maximum_prefill_batch;
    uint64_t maximum_decode_batch;
    double prefill_tokens_per_second;
    double decode_tokens_per_second;
    double average_ttft_ms;
    double average_itl_ms;
    double scheduler_ms;
    uint64_t chunked_prefill_steps;
    uint64_t chunked_prefill_tokens;
    uint64_t maximum_prefill_chunk;
    uint64_t prefix_cache_hits;
    uint64_t prefix_cache_misses;
    uint64_t prefix_cache_partial_hits;
    uint64_t prefix_cache_inserts;
    uint64_t prefix_cache_evictions;
    uint64_t prefix_reused_tokens;
    uint64_t prefix_cow_pages;
    uint64_t prefix_cow_bytes;
    uint64_t attention_parallel_calls;
} lfm25_cpu_engine_metrics_v3;

LFM25_CPU_API uint32_t lfm25_cpu_api_version(void);
LFM25_CPU_API void lfm25_cpu_model_options_init(lfm25_cpu_model_options_v6* options);
LFM25_CPU_API void lfm25_cpu_generation_options_init(lfm25_cpu_generation_options_v1* options);
LFM25_CPU_API void lfm25_cpu_engine_options_init(lfm25_cpu_engine_options_v3* options);
LFM25_CPU_API void lfm25_cpu_request_options_init(lfm25_cpu_request_options_v1* options);
LFM25_CPU_API const char* lfm25_cpu_detected_isa(void);
LFM25_CPU_API const char* lfm25_cpu_capabilities(void);
LFM25_CPU_API const char* lfm25_cpu_topology(void);

LFM25_CPU_API lfm25_cpu_model* lfm25_cpu_model_create(
    const char* safetensors_path,
    const lfm25_cpu_model_options_v6* options,
    const lfm25_cpu_generation_options_v1* generation);
LFM25_CPU_API void lfm25_cpu_model_destroy(lfm25_cpu_model* model);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_prefill(
    lfm25_cpu_model* model, const int32_t* tokens, size_t token_count);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_decode(
    lfm25_cpu_model* model, int32_t* token);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_copy_logits(
    lfm25_cpu_model* model, float* output, size_t capacity, size_t* required);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_get_metrics(
    lfm25_cpu_model* model, lfm25_cpu_runtime_metrics_v1* metrics);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_get_memory_stats(
    lfm25_cpu_model* model, lfm25_cpu_memory_stats_v2* stats);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_model_vocab_size(
    const lfm25_cpu_model* model, int32_t* vocab_size);
LFM25_CPU_API const char* lfm25_cpu_model_backend_description(lfm25_cpu_model* model);
LFM25_CPU_API const char* lfm25_cpu_model_pack_path(lfm25_cpu_model* model);
LFM25_CPU_API const char* lfm25_cpu_model_last_error(lfm25_cpu_model* model);

LFM25_CPU_API lfm25_cpu_engine* lfm25_cpu_engine_create(
    const char* safetensors_path,
    const lfm25_cpu_model_options_v6* model_options,
    const lfm25_cpu_engine_options_v3* engine_options);
LFM25_CPU_API void lfm25_cpu_engine_destroy(lfm25_cpu_engine* engine);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_submit(
    lfm25_cpu_engine* engine, const int32_t* tokens, size_t token_count,
    const lfm25_cpu_request_options_v1* options,
    lfm25_cpu_request_id* request_id);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_poll(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id,
    int32_t* output, size_t capacity, size_t* token_count, int* finished);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_request_status(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id,
    int32_t* status);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_cancel(
    lfm25_cpu_engine* engine, lfm25_cpu_request_id request_id);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_step(
    lfm25_cpu_engine* engine, int* progressed);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_start(lfm25_cpu_engine* engine);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_stop(lfm25_cpu_engine* engine);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_engine_get_metrics(
    lfm25_cpu_engine* engine, lfm25_cpu_engine_metrics_v3* metrics);
LFM25_CPU_API const char* lfm25_cpu_engine_backend_description(lfm25_cpu_engine* engine);
LFM25_CPU_API const char* lfm25_cpu_engine_last_error(lfm25_cpu_engine* engine);

LFM25_CPU_API lfm25_cpu_tokenizer* lfm25_cpu_tokenizer_create(const char* tokenizer_json_path);
LFM25_CPU_API void lfm25_cpu_tokenizer_destroy(lfm25_cpu_tokenizer* tokenizer);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_tokenizer_encode(
    lfm25_cpu_tokenizer* tokenizer, const char* text, int add_bos,
    int32_t* output, size_t capacity, size_t* required);
LFM25_CPU_API lfm25_cpu_status lfm25_cpu_tokenizer_decode(
    lfm25_cpu_tokenizer* tokenizer, const int32_t* tokens, size_t token_count,
    int skip_special, char* output, size_t capacity, size_t* required);
LFM25_CPU_API const char* lfm25_cpu_tokenizer_last_error(lfm25_cpu_tokenizer* tokenizer);

#ifdef __cplusplus
}
#endif
#endif
