#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace lfm {

void launch_embedding(int32_t token, const __nv_bfloat16* table, __nv_bfloat16* out,
                      int hidden, cudaStream_t stream);
void launch_embedding_device(const int32_t* token, const __nv_bfloat16* table,
                             __nv_bfloat16* out, int hidden, cudaStream_t stream);
void launch_embedding_batch(const int32_t* tokens, int rows,
                            const __nv_bfloat16* table, __nv_bfloat16* out,
                            int hidden, cudaStream_t stream);

void launch_embedding_int8(int32_t token, const int8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream);
void launch_embedding_int8_device(const int32_t* token, const int8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream);
void launch_embedding_int8_batch(const int32_t* tokens, int rows,
                                 const int8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream);

void launch_w8a16_linear(const __nv_bfloat16* x, const int8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream);

void launch_embedding_int4(int32_t token, const uint8_t* table,
                           const float* scales, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream);
void launch_embedding_int4_device(const int32_t* token, const uint8_t* table,
                                  const float* scales, __nv_bfloat16* out,
                                  int hidden, cudaStream_t stream);
void launch_embedding_int4_batch(const int32_t* tokens, int rows,
                                 const uint8_t* table, const float* scales,
                                 __nv_bfloat16* out, int hidden,
                                 cudaStream_t stream);

void launch_w4a16_linear(const __nv_bfloat16* x, const uint8_t* weight,
                         const float* scales, __nv_bfloat16* y,
                         int m, int n, int k, float beta,
                         cudaStream_t stream);

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    __nv_bfloat16* out, int rows, int width, float eps,
                    cudaStream_t stream);
void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                         int count, cudaStream_t stream);
void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                         int count, cudaStream_t stream);

void launch_conv_decode(const __nv_bfloat16* projected_bcx,
                        const __nv_bfloat16* conv_weight,
                        __nv_bfloat16* state, __nv_bfloat16* y,
                        int hidden, int cache_length, int position,
                        cudaStream_t stream);
void launch_conv_decode_device(const __nv_bfloat16* projected_bcx,
                               const __nv_bfloat16* conv_weight,
                               __nv_bfloat16* state, __nv_bfloat16* y,
                               int hidden, int cache_length,
                               const int32_t* position,
                               cudaStream_t stream);
void launch_conv_prefill(const __nv_bfloat16* projected_bcx,
                         const __nv_bfloat16* conv_weight,
                         __nv_bfloat16* state, __nv_bfloat16* y,
                         int rows, int hidden, int cache_length,
                         cudaStream_t stream);

void launch_qk_norm_rope_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                const __nv_bfloat16* q_norm,
                                const __nv_bfloat16* k_norm,
                                const __nv_bfloat16* rope_cos,
                                const __nv_bfloat16* rope_sin,
                                int q_heads, int kv_heads, int head_dim,
                                int position, float eps,
                                cudaStream_t stream);
void launch_qk_norm_rope_strict_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                       const __nv_bfloat16* q_norm,
                                       const __nv_bfloat16* k_norm,
                                       const __nv_bfloat16* rope_cos,
                                       const __nv_bfloat16* rope_sin,
                                       int q_heads, int kv_heads, int head_dim,
                                       const int32_t* position, float eps,
                                       cudaStream_t stream);
void launch_qk_norm_rope_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                              const __nv_bfloat16* q_norm,
                              const __nv_bfloat16* k_norm,
                              const __nv_bfloat16* rope_cos,
                              const __nv_bfloat16* rope_sin,
                              int q_heads, int kv_heads, int head_dim,
                              int position, float eps,
                              cudaStream_t stream);
void launch_qk_norm_rope_fast_device(__nv_bfloat16* q, __nv_bfloat16* k,
                                     const __nv_bfloat16* q_norm,
                                     const __nv_bfloat16* k_norm,
                                     const __nv_bfloat16* rope_cos,
                                     const __nv_bfloat16* rope_sin,
                                     int q_heads, int kv_heads, int head_dim,
                                     const int32_t* position, float eps,
                                     cudaStream_t stream);
void launch_qk_norm_rope_prefill_strict(__nv_bfloat16* q, __nv_bfloat16* k,
                                        const __nv_bfloat16* q_norm,
                                        const __nv_bfloat16* k_norm,
                                        const __nv_bfloat16* rope_cos,
                                        const __nv_bfloat16* rope_sin,
                                        int rows, int q_heads, int kv_heads,
                                        int head_dim, float eps,
                                        cudaStream_t stream);
void launch_qk_norm_rope_prefill_fast(__nv_bfloat16* q, __nv_bfloat16* k,
                                      const __nv_bfloat16* q_norm,
                                      const __nv_bfloat16* k_norm,
                                      const __nv_bfloat16* rope_cos,
                                      const __nv_bfloat16* rope_sin,
                                      int rows, int q_heads, int kv_heads,
                                      int head_dim, float eps,
                                      cudaStream_t stream);

void launch_store_kv(const __nv_bfloat16* k, const __nv_bfloat16* v,
                     __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
                     int position, int kv_width, cudaStream_t stream);
void launch_store_kv_device(const __nv_bfloat16* k, const __nv_bfloat16* v,
                            __nv_bfloat16* key_cache,
                            __nv_bfloat16* value_cache,
                            const int32_t* position, int kv_width,
                            cudaStream_t stream);
void launch_store_kv_prefill(const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* key_cache,
                             __nv_bfloat16* value_cache,
                             int rows, int kv_width, cudaStream_t stream);

void launch_store_kv_int8(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          int8_t* key_cache, int8_t* value_cache,
                          float* key_scales, float* value_scales,
                          int position, int kv_heads, int head_dim,
                          cudaStream_t stream);
void launch_store_kv_int8_device(const __nv_bfloat16* k,
                                 const __nv_bfloat16* v,
                                 int8_t* key_cache, int8_t* value_cache,
                                 float* key_scales, float* value_scales,
                                 const int32_t* position, int kv_heads,
                                 int head_dim, cudaStream_t stream);
void launch_store_kv_int8_prefill(const __nv_bfloat16* k,
                                  const __nv_bfloat16* v,
                                  int8_t* key_cache, int8_t* value_cache,
                                  float* key_scales, float* value_scales,
                                  int rows, int kv_heads, int head_dim,
                                  cudaStream_t stream);

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

void launch_argmax_bf16(const __nv_bfloat16* logits, int count,
                        int32_t* result, cudaStream_t stream);
void launch_mark_seen_batch(const int32_t* tokens, int count,
                            uint8_t* seen, int vocab, cudaStream_t stream);
void launch_mark_seen_batch_ptrs(const int32_t* tokens,
                                 uint8_t* const* seen,
                                 int rows, int vocab,
                                 cudaStream_t stream);
void launch_mark_seen(const int32_t* token, uint8_t* seen, int vocab,
                      cudaStream_t stream);
void launch_prepare_sampling_scores(const __nv_bfloat16* logits,
                                    const uint8_t* seen,
                                    float* scores, int vocab,
                                    float temperature,
                                    float repetition_penalty,
                                    cudaStream_t stream);
void launch_select_topk(float* scores, float* selected_values,
                        int32_t* selected_indices, int rank, int vocab,
                        cudaStream_t stream);
void launch_sample_topk(const float* selected_values,
                        const int32_t* selected_indices,
                        int top_k, float top_p, uint64_t* rng_state,
                        int32_t* result, cudaStream_t stream);
void launch_fused_sample_topk(const __nv_bfloat16* logits,
                              uint8_t* seen,
                              float* scores,
                              float* selected_values,
                              int32_t* selected_indices,
                              int vocab,
                              float temperature,
                              float repetition_penalty,
                              int top_k,
                              float top_p,
                              uint64_t* rng_state,
                              int32_t* result,
                              cudaStream_t stream);

// Packed multi-session decode helpers. Pointer arrays contain one persistent
// session allocation per packed row; activations stay contiguous by batch.
void launch_packed_sample_topk(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k,
    const float* top_p,
    float* scores, float* selected_values,
    int32_t* selected_indices,
    int rows, int vocab, int32_t* result,
    cudaStream_t stream);

void launch_split_qkv_rows(const __nv_bfloat16* qkv,
                           __nv_bfloat16* q,
                           __nv_bfloat16* k,
                           __nv_bfloat16* v,
                           int rows, int q_width, int kv_width,
                           cudaStream_t stream);
void launch_swiglu_interleaved(const __nv_bfloat16* gate_up,
                               __nv_bfloat16* out,
                               int rows, int intermediate,
                               cudaStream_t stream);

void launch_qk_norm_rope_batch_positions(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm,
    const __nv_bfloat16* k_norm,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    const int32_t* positions, int rows,
    int q_heads, int kv_heads, int head_dim,
    float eps, bool fast, cudaStream_t stream);

void launch_store_kv_batch_ptrs(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* const* key_cache,
    __nv_bfloat16* const* value_cache,
    const int32_t* positions, int rows, int kv_width,
    cudaStream_t stream);
void launch_store_kv_int8_batch_ptrs(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* const* key_cache, int8_t* const* value_cache,
    float* const* key_scales, float* const* value_scales,
    const int32_t* positions, int rows,
    int kv_heads, int head_dim, cudaStream_t stream);

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

void launch_conv_decode_batch_ptrs(
    const __nv_bfloat16* projected_bcx,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    int rows, int hidden, int cache_length,
    cudaStream_t stream);

void launch_scatter_bf16_rows(
    const __nv_bfloat16* source,
    __nv_bfloat16* const* destinations,
    int rows, int width, cudaStream_t stream);
void launch_scatter_decode_state(
    const int32_t* sampled,
    const int32_t* positions,
    int32_t* const* sampled_destinations,
    int32_t* const* position_destinations,
    int rows, cudaStream_t stream);

void launch_increment_position(int32_t* position, cudaStream_t stream);

// Physically paged KV cache used by the concurrent packed decoder. page_tables
// is [rows, page_table_stride] and contains physical page IDs.
void launch_store_kv_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream);
void launch_store_kv_int8_paged_batch(
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    int8_t* key_pool, int8_t* value_pool,
    float* key_scale_pool, float* value_scale_pool,
    const uint32_t* page_tables, int page_table_stride,
    const int32_t* positions, int rows, int attention_slot,
    int page_tokens, int attention_layers, int kv_heads, int head_dim,
    cudaStream_t stream);
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

} // namespace lfm
