#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   __nv_bfloat16* out, int rows, int width, float eps,
                   cudaStream_t stream);
void launch_residual_add(__nv_bfloat16* x, const __nv_bfloat16* residual,
                        int count, cudaStream_t stream);
void launch_scale(__nv_bfloat16* x, int count, float scale,
                  cudaStream_t stream);
void launch_tanh_softcap(__nv_bfloat16* x, int count, float cap,
                         cudaStream_t stream);
void launch_swiglu_fused(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                        int count, cudaStream_t stream);
void launch_relu2(const __nv_bfloat16* input, __nv_bfloat16* out,
                  int count, cudaStream_t stream);
void launch_gelu_tanh(const __nv_bfloat16* input, __nv_bfloat16* out,
                      int count, cudaStream_t stream);
void launch_gated_gelu_tanh(const __nv_bfloat16* gate_up, __nv_bfloat16* out,
                            int count, cudaStream_t stream);
void launch_multiply(__nv_bfloat16* x, const __nv_bfloat16* y, int count,
                     cudaStream_t stream);
void launch_sigmoid_multiply(__nv_bfloat16* x, const __nv_bfloat16* gate,
                             int count, cudaStream_t stream);
void launch_extract_attention_output_gate(const __nv_bfloat16* packed,
                               __nv_bfloat16* query_out, __nv_bfloat16* gate_out,
                               int rows, int width, int head_dim, cudaStream_t stream);
void launch_multiply_strided(__nv_bfloat16* x, const __nv_bfloat16* y,
                             int rows, int width, int y_stride, int y_offset,
                             cudaStream_t stream);
void launch_scale_by_scalar(__nv_bfloat16* x, const __nv_bfloat16* scalar,
                            int count, cudaStream_t stream);
void launch_sigmoid_scale_by_scalar(__nv_bfloat16* x, const __nv_bfloat16* scalar,
                                    int count, cudaStream_t stream);
void launch_sigmoid_multiply_strided(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                     int rows, int width, cudaStream_t stream);
void launch_sigmoid_multiply_headwise(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                      int rows, int heads, int head_dim,
                                      cudaStream_t stream);

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
void launch_conv_ragged_prefill(
    const __nv_bfloat16* projected_bcx,
    const __nv_bfloat16* conv_weight,
    __nv_bfloat16* const* states,
    __nv_bfloat16* y,
    const int32_t* positions,
    const int32_t* span_offsets,
    const int32_t* span_counts,
    int requests, int hidden, int cache_length,
    cudaStream_t stream);

}
