#ifndef LFM25_C_API_H
#define LFM25_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LFM25_C_API_BUILD)
#define LFM25_API __declspec(dllexport)
#elif defined(_WIN32)
#define LFM25_API __declspec(dllimport)
#else
#define LFM25_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LFM25_C_API_VERSION 6u

typedef struct lfm25_model lfm25_model;
typedef struct lfm25_tokenizer lfm25_tokenizer;

typedef struct lfm25_engine lfm25_engine;
typedef uint64_t lfm25_request_id;
typedef struct lfm25_model_options_v2 lfm25_model_options_v2;
typedef struct lfm25_generation_options_v1 lfm25_generation_options_v1;

typedef enum lfm25_request_status {
    LFM25_REQUEST_QUEUED = 0,
    LFM25_REQUEST_PREFILL = 1,
    LFM25_REQUEST_DECODING = 2,
    LFM25_REQUEST_FINISHED = 3,
    LFM25_REQUEST_CANCELLED = 4,
    LFM25_REQUEST_FAILED = 5
} lfm25_request_status;

typedef enum lfm25_scheduler_policy {
    LFM25_SCHEDULER_GUARANTEED_NO_EVICT = 0,
    LFM25_SCHEDULER_MAX_UTILIZATION = 1
} lfm25_scheduler_policy;

typedef enum lfm25_status {
    LFM25_STATUS_OK = 0,
    LFM25_STATUS_INVALID_ARGUMENT = 1,
    LFM25_STATUS_RUNTIME_ERROR = 2,
    LFM25_STATUS_STOPPED = 3,
    LFM25_STATUS_BUFFER_TOO_SMALL = 4
} lfm25_status;

typedef enum lfm25_weight_mode {
    LFM25_WEIGHT_BF16 = 0,
    LFM25_WEIGHT_INT8 = 1,
    LFM25_WEIGHT_INT4 = 2
} lfm25_weight_mode;

typedef enum lfm25_kv_cache_mode {
    LFM25_KV_BF16 = 0,
    LFM25_KV_INT8 = 1
} lfm25_kv_cache_mode;

typedef enum lfm25_gemm_backend {
    LFM25_GEMM_CUBLAS = 0,
    LFM25_GEMM_CUBLASLT = 1
} lfm25_gemm_backend;

typedef enum lfm25_attention_mode {
    LFM25_ATTENTION_SINGLE = 0,
    LFM25_ATTENTION_SEGMENTED = 1,
    LFM25_ATTENTION_AUTO = 2
} lfm25_attention_mode;

enum {
    LFM25_OPTION_FUSED_RESIDUALS = 1u << 0,
    LFM25_OPTION_FAST_ATTENTION = 1u << 1,
    LFM25_OPTION_FUSED_PROJECTIONS = 1u << 2,
    LFM25_OPTION_FUSED_SAMPLING = 1u << 3,
    LFM25_OPTION_CUDA_GRAPH = 1u << 4,
    LFM25_OPTION_LEGACY_PREFILL = 1u << 5,
    LFM25_OPTION_LT_AUTOTUNE = 1u << 6
};

typedef struct lfm25_model_options_v2 {
    uint32_t struct_size;
    uint32_t flags;
    int32_t max_context;
    int32_t weight_mode;
    int32_t kv_cache_mode;
    int32_t gemm_backend;
    int32_t attention_mode;
    int32_t attention_chunk_tokens;
    int32_t attention_auto_threshold;
    int32_t lt_workspace_mb;
    int32_t lt_heuristics;

    int32_t expert_offload_mode;
    int32_t expert_host_mode;
    int32_t expert_cache_policy;
    uint64_t gpu_expert_cache_bytes;
    int32_t experts_per_layer;
    uint64_t maximum_pinned_host_bytes;
    uint64_t gpu_memory_reserve_bytes;
    int32_t prefill_chunk_tokens;
    int32_t prefetch_experts;

    int32_t expert_backing;
    uint64_t host_expert_cache_bytes;
    int32_t expert_io_backend;
    int32_t expert_io_queue_depth;
    int32_t expert_io_workers;
    int32_t expert_direct_io;
    const char* expert_sidecar_path;
    const char* mirror_path;
    const char* usage_profile_path;
} lfm25_model_options_v2;

typedef struct lfm25_generation_options_v1 {
    uint32_t struct_size;
    float temperature;
    int32_t top_k;
    float top_p;
    float repetition_penalty;
    uint64_t seed;
} lfm25_generation_options_v1;

typedef struct lfm25_engine_options_v3 {
    uint32_t struct_size;
    int32_t max_context;
    int32_t max_active_requests;
    int32_t max_batched_tokens;
    int32_t prefill_chunk_tokens;
    int32_t page_tokens;
    uint64_t logical_kv_pages;
    int32_t scheduler_policy;
    int32_t worker_thread;
    int32_t idle_sleep_microseconds;
    lfm25_model_options_v2 model;
} lfm25_engine_options_v3;

typedef struct lfm25_request_options_v2 {
    uint32_t struct_size;
    int32_t max_new_tokens;
    int32_t eos_token;
    int32_t priority;
    lfm25_generation_options_v1 generation;
} lfm25_request_options_v2;

typedef struct lfm25_engine_metrics_v2 {
    uint32_t struct_size;
    uint64_t submitted;
    uint64_t completed;
    uint64_t cancelled;
    uint64_t failed;
    uint64_t scheduler_steps;
    uint64_t prefill_tokens;
    uint64_t decoded_tokens;
    uint64_t active_requests;
    uint64_t queued_requests;
    uint64_t logical_pages_used;
    uint64_t logical_pages_total;
    uint64_t ttft_samples;
    uint64_t itl_samples;
    double cumulative_ttft_ms;
    double cumulative_itl_ms;
    double average_ttft_ms;
    double average_itl_ms;
    double cumulative_step_ms;
} lfm25_engine_metrics_v2;


typedef struct lfm25_packed_metrics_v1 {
    uint32_t struct_size;
    uint64_t packed_decode_steps;
    uint64_t packed_decode_tokens;
    uint64_t lane_decode_tokens;
    uint64_t packed_fallback_batches;
    uint64_t maximum_packed_batch;
    double cumulative_packed_decode_ms;
    double packed_tokens_per_second;
} lfm25_packed_metrics_v1;

typedef struct lfm25_packed_metrics_v2 {
    uint32_t struct_size;
    uint64_t packed_decode_steps;
    uint64_t packed_decode_tokens;
    uint64_t lane_decode_tokens;
    uint64_t packed_fallback_batches;
    uint64_t maximum_packed_batch;
    double cumulative_packed_decode_ms;
    double packed_tokens_per_second;
    uint64_t ragged_prefill_steps;
    uint64_t ragged_prefill_tokens;
    uint64_t lane_prefill_tokens;
    uint64_t maximum_ragged_prefill_batch;
    double cumulative_ragged_prefill_ms;
    double ragged_prefill_tokens_per_second;
} lfm25_packed_metrics_v2;

typedef struct lfm25_paged_kv_metrics_v1 {
    uint32_t struct_size;
    uint64_t physical_pages_used;
    uint64_t physical_pages_total;
    uint64_t physical_kv_bytes;
    uint64_t prefix_cache_hits;
    uint64_t prefix_cache_misses;
    uint64_t prefix_cache_inserts;
    uint64_t prefix_cache_evictions;
} lfm25_paged_kv_metrics_v1;

typedef struct lfm25_paged_kv_metrics_v2 {
    uint32_t struct_size;
    uint64_t physical_pages_used;
    uint64_t physical_pages_total;
    uint64_t physical_kv_bytes;
    uint64_t prefix_cache_hits;
    uint64_t prefix_cache_misses;
    uint64_t prefix_cache_inserts;
    uint64_t prefix_cache_evictions;
    uint64_t prefix_cache_partial_hits;
    uint64_t prefix_reused_tokens;
    uint64_t prefix_cow_pages;
    uint64_t direct_paged_prefill_tokens;
    uint64_t segmented_paged_decode_steps;
    uint64_t segmented_paged_decode_tokens;
} lfm25_paged_kv_metrics_v2;

typedef struct lfm25_paged_kv_metrics_v3 {
    uint32_t struct_size;
    uint64_t physical_pages_used;
    uint64_t physical_pages_total;
    uint64_t physical_kv_bytes;
    uint64_t prefix_cache_hits;
    uint64_t prefix_cache_misses;
    uint64_t prefix_cache_inserts;
    uint64_t prefix_cache_evictions;
    uint64_t prefix_cache_partial_hits;
    uint64_t prefix_reused_tokens;
    uint64_t prefix_cow_pages;
    uint64_t direct_paged_prefill_tokens;
    uint64_t segmented_paged_decode_steps;
    uint64_t segmented_paged_decode_tokens;
    uint64_t prefix_radix_nodes;
    uint64_t prefix_radix_lookups;
    uint64_t prefix_cow_bytes_copied;
    uint64_t prefix_cow_bytes_saved;
} lfm25_paged_kv_metrics_v3;

typedef struct lfm25_memory_stats_v1 {
    uint32_t struct_size;
    uint64_t weights;
    uint64_t kv_cache;
    uint64_t conv_state;
    uint64_t rope_tables;
    uint64_t activations;
    uint64_t sampling;
    uint64_t matmul_workspace;
    uint64_t attention_workspace;
    uint64_t total;
} lfm25_memory_stats_v1;

typedef struct lfm25_runtime_metrics_v1 {
    uint32_t struct_size;
    double last_prefill_ms;
    uint64_t prefill_tokens;
    double prefill_tokens_per_second;
    double cumulative_decode_ms;
    uint64_t decoded_tokens;
    double decode_tokens_per_second;
} lfm25_runtime_metrics_v1;

typedef int (*lfm25_token_callback)(int32_t token, void* user_data);

LFM25_API uint32_t lfm25_api_version(void);
LFM25_API void lfm25_model_options_v2_init(lfm25_model_options_v2* options);
LFM25_API void lfm25_generation_options_init(lfm25_generation_options_v1* options);

LFM25_API lfm25_model* lfm25_model_create(
    const char* safetensors_path,
    const lfm25_model_options_v2* model_options,
    const lfm25_generation_options_v1* generation_options);
LFM25_API void lfm25_model_destroy(lfm25_model* model);
LFM25_API const char* lfm25_last_error(const lfm25_model* model);

LFM25_API lfm25_status lfm25_reset(lfm25_model* model);
LFM25_API lfm25_status lfm25_prefill(
    lfm25_model* model, const int32_t* tokens, size_t token_count);
LFM25_API lfm25_status lfm25_decode(lfm25_model* model, int32_t* token);
LFM25_API lfm25_status lfm25_position(
    const lfm25_model* model, int32_t* position);
LFM25_API lfm25_status lfm25_model_vocab_size(
    const lfm25_model* model, int32_t* vocab_size);
LFM25_API lfm25_status lfm25_copy_logits(
    lfm25_model* model, float* logits, size_t logits_count);
LFM25_API lfm25_status lfm25_generate(
    lfm25_model* model, int32_t max_new_tokens, int32_t eos_token,
    lfm25_token_callback callback, void* user_data, int32_t* generated_tokens);

LFM25_API lfm25_status lfm25_save_session(
    lfm25_model* model, const char* path);
LFM25_API lfm25_status lfm25_load_session(
    lfm25_model* model, const char* path);
LFM25_API lfm25_status lfm25_get_memory_stats(
    const lfm25_model* model, lfm25_memory_stats_v1* stats);
LFM25_API lfm25_status lfm25_get_runtime_metrics(
    const lfm25_model* model, lfm25_runtime_metrics_v1* metrics);

LFM25_API void lfm25_engine_options_v3_init(lfm25_engine_options_v3* options);
LFM25_API void lfm25_request_options_init(lfm25_request_options_v2* options);
LFM25_API lfm25_engine* lfm25_engine_create(
    const char* safetensors_path, const lfm25_engine_options_v3* options);
LFM25_API void lfm25_engine_destroy(lfm25_engine* engine);
LFM25_API const char* lfm25_engine_last_error(const lfm25_engine* engine);
LFM25_API lfm25_status lfm25_engine_start(lfm25_engine* engine);
LFM25_API lfm25_status lfm25_engine_stop(lfm25_engine* engine);
LFM25_API lfm25_status lfm25_engine_step(
    lfm25_engine* engine, int* did_work);
LFM25_API lfm25_status lfm25_engine_submit(
    lfm25_engine* engine, const int32_t* prompt_tokens, size_t prompt_count,
    const lfm25_request_options_v2* options, lfm25_request_id* request_id);
LFM25_API lfm25_status lfm25_engine_cancel(
    lfm25_engine* engine, lfm25_request_id request_id);
LFM25_API lfm25_status lfm25_engine_request_status(
    const lfm25_engine* engine, lfm25_request_id request_id,
    int32_t* request_status);
LFM25_API lfm25_status lfm25_engine_poll(
    lfm25_engine* engine, lfm25_request_id request_id,
    int32_t* tokens, size_t token_capacity, size_t* token_count,
    int32_t* request_status, int* finished);
LFM25_API lfm25_status lfm25_engine_get_metrics(
    const lfm25_engine* engine, lfm25_engine_metrics_v2* metrics);
LFM25_API lfm25_status lfm25_engine_get_packed_metrics(
    const lfm25_engine* engine, lfm25_packed_metrics_v1* metrics);
LFM25_API lfm25_status lfm25_engine_get_packed_metrics_v2(
    const lfm25_engine* engine, lfm25_packed_metrics_v2* metrics);
LFM25_API lfm25_status lfm25_engine_get_paged_kv_metrics(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v1* metrics);
LFM25_API lfm25_status lfm25_engine_get_paged_kv_metrics_v2(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v2* metrics);
LFM25_API lfm25_status lfm25_engine_get_paged_kv_metrics_v3(
    const lfm25_engine* engine, lfm25_paged_kv_metrics_v3* metrics);

LFM25_API lfm25_tokenizer* lfm25_tokenizer_create(
    const char* tokenizer_json_path);
LFM25_API void lfm25_tokenizer_destroy(lfm25_tokenizer* tokenizer);
LFM25_API const char* lfm25_tokenizer_last_error(
    const lfm25_tokenizer* tokenizer);
LFM25_API int32_t lfm25_tokenizer_bos_id(const lfm25_tokenizer* tokenizer);
LFM25_API int32_t lfm25_tokenizer_eos_id(const lfm25_tokenizer* tokenizer);
LFM25_API lfm25_status lfm25_tokenizer_encode(
    lfm25_tokenizer* tokenizer, const char* text, int add_bos,
    int32_t* token_ids, size_t token_capacity, size_t* required_tokens);
LFM25_API lfm25_status lfm25_tokenizer_decode(
    lfm25_tokenizer* tokenizer, const int32_t* token_ids, size_t token_count,
    int skip_special, char* text, size_t text_capacity, size_t* required_bytes);
LFM25_API lfm25_status lfm25_tokenizer_format_chat(
    lfm25_tokenizer* tokenizer, const char* user_prompt,
    const char* system_prompt, char* text, size_t text_capacity,
    size_t* required_bytes);

#ifdef __cplusplus
}
#endif

#endif
