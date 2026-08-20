#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_embedding(int32_t token, const __nv_bfloat16* table, __nv_bfloat16* out,
                     int hidden, cudaStream_t stream);
void launch_embedding_device(const int32_t* token, const __nv_bfloat16* table,
                            __nv_bfloat16* out, int hidden, cudaStream_t stream);
void launch_embedding_batch(const int32_t* tokens, int rows,
                           const __nv_bfloat16* table, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream);
void launch_embedding_slice(int32_t token, const __nv_bfloat16* table,
                            int table_width, int offset, __nv_bfloat16* out,
                            int width, cudaStream_t stream);
void launch_embedding_slice_device(const int32_t* token, const __nv_bfloat16* table,
                                   int table_width, int offset, __nv_bfloat16* out,
                                   int width, cudaStream_t stream);
void launch_embedding_slice_batch(const int32_t* tokens, int rows,
                                  const __nv_bfloat16* table, int table_width,
                                  int offset, __nv_bfloat16* out, int width,
                                  cudaStream_t stream);

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

// Dynamic per-row FP8 E4M3 quantization; used both for per-token activation
// quantization and per-channel weight quantization (see linear.cuh).
void launch_quantize_e4m3_per_row(const __nv_bfloat16* x, __nv_fp8_e4m3* q,
                                 float* scales, int rows, int k,
                                 cudaStream_t stream);

// Applies the W8A8 outer-product dequant scale (act_scale[row] *
// weight_scale[col]) to a raw, unscaled FP8xFP8->FP32 matmul accumulation.
void launch_fp8_scale_apply(const float* raw, const float* act_scale,
                           const float* weight_scale, __nv_bfloat16* y,
                           int m, int n, float beta, cudaStream_t stream);

// Naive (non-cuBLASLt) FP8 W8A8 matmul, correct for any m/n/k. Fallback for
// shapes the fp8 cuBLASLt heuristic can't find an algorithm for.
void launch_fp8_w8a8_naive(const __nv_fp8_e4m3* x_q, const float* act_scale,
                          const __nv_fp8_e4m3* w_q, const float* weight_scale,
                          __nv_bfloat16* y, int m, int n, int k, float beta,
                          cudaStream_t stream);

}
