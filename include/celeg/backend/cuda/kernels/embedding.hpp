#pragma once

#include <cuda_bf16.h>
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

}
