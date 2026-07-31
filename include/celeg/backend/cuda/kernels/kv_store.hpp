#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

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

} // namespace celeg
