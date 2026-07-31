#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_gqa_decode_strict(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream);
void launch_gqa_decode_strict_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream);
void launch_gqa_decode_online(const __nv_bfloat16* q,
                              const __nv_bfloat16* key_cache,
                              const __nv_bfloat16* value_cache,
                              __nv_bfloat16* out, int seq_len,
                              int q_heads, int kv_heads, int head_dim,
                              cudaStream_t stream);
void launch_gqa_decode_online_device(const __nv_bfloat16* q,
                                     const __nv_bfloat16* key_cache,
                                     const __nv_bfloat16* value_cache,
                                     __nv_bfloat16* out,
                                     const int32_t* position,
                                     int q_heads, int kv_heads, int head_dim,
                                     cudaStream_t stream);

void launch_gqa_decode_segmented_device(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream);
void launch_gqa_prefill_strict(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream);
void launch_gqa_prefill_online(const __nv_bfloat16* q,
                               const __nv_bfloat16* key_cache,
                               const __nv_bfloat16* value_cache,
                               __nv_bfloat16* out, int rows,
                               int q_heads, int kv_heads, int head_dim,
                               cudaStream_t stream);

// Chunked/segmented causal prefill attention. Each (row, head, chunk) block
// handles at most `chunk_tokens` KV positions (causally clamped to the row's
// own position) instead of the single serial per-row loop in
// launch_gqa_prefill_online, then a reduce kernel combines the chunk-local
// online-softmax partials per row/head. Trades one extra kernel + O(rows *
// q_heads * chunks) scratch for far more block-level parallelism and a much
// shorter per-block critical path. `chunks` must equal
// ceil(rows / chunk_tokens); `partial_max`/`partial_denom` must hold
// rows*q_heads*chunks floats and `partial_accum`
// rows*q_heads*chunks*head_dim floats.
void launch_gqa_prefill_segmented(
    const __nv_bfloat16* q, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream);

// Batched-GEMM causal prefill attention (see the private attention .cuh
// implementation for the
// rationale). `scores_scratch` needs q_heads*rows*rows floats,
// `probs_scratch` the same element count in BF16. `q_width`/`kv_width` are
// the interleaved-head row strides of `q`/`k`/`v`; `out_width` is the row
// stride of `out` (the attention-output buffer consumed by the out-proj
// linear layer).
void launch_gqa_prefill_gemm(
    cublasHandle_t cublas, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* out, float* scores_scratch,
    __nv_bfloat16* probs_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int q_width, int kv_width, int out_width,
    cudaStream_t stream);

// Flash attention prefill (tiled online-softmax, no score matrix materialization).
void launch_gqa_prefill_flash(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim,
    int q_width, int kv_width, int out_width,
    cudaStream_t stream);

void launch_gqa_decode_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream);
void launch_gqa_decode_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int seq_len,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream);

void launch_gqa_decode_strict_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream);
void launch_gqa_decode_online_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream);
void launch_gqa_decode_segmented_int8_device(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out,
    const int32_t* position, int q_heads, int kv_heads, int head_dim,
    int chunk_tokens, int chunks, float* partial_max,
    float* partial_denom, float* partial_accum, cudaStream_t stream);
void launch_gqa_prefill_strict_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream);
void launch_gqa_prefill_online_int8(
    const __nv_bfloat16* q, const int8_t* key_cache,
    const int8_t* value_cache, const float* key_scales,
    const float* value_scales, __nv_bfloat16* out, int rows,
    int q_heads, int kv_heads, int head_dim, cudaStream_t stream);

void launch_gqa_decode_batch_ptrs(
    const __nv_bfloat16* q,
    const __nv_bfloat16* const* key_cache,
    const __nv_bfloat16* const* value_cache,
    __nv_bfloat16* out, const int32_t* positions,
    int rows, int q_heads, int kv_heads, int head_dim,
    bool fast, cudaStream_t stream);
void launch_gqa_decode_int8_batch_ptrs(
    const __nv_bfloat16* q,
    const int8_t* const* key_cache,
    const int8_t* const* value_cache,
    const float* const* key_scales,
    const float* const* value_scales,
    __nv_bfloat16* out, const int32_t* positions,
    int rows, int q_heads, int kv_heads, int head_dim,
    bool fast, cudaStream_t stream);

void launch_gqa_decode_paged_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream);
void launch_gqa_decode_int8_paged_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, bool fast,
    cudaStream_t stream);
void launch_gqa_decode_paged_segmented_batch(
    const __nv_bfloat16* q,
    const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream);
void launch_gqa_decode_int8_paged_segmented_batch(
    const __nv_bfloat16* q,
    const int8_t* key_pool, const int8_t* value_pool,
    const float* key_scale_pool, const float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    __nv_bfloat16* out, const int32_t* positions, int rows,
    int attention_slot, int page_tokens, int attention_layers,
    int q_heads, int kv_heads, int head_dim, int chunk_tokens, int chunks,
    float* partial_max, float* partial_denom, float* partial_accum,
    cudaStream_t stream);

} // namespace celeg
