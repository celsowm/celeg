#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "utils.cuh"

#include <celeg/checkpoint/formats/gguf.hpp>
#include "detail/linear_weights.hpp"

namespace celeg {

bool cuda_gguf_native_mmq(GgmlType type);

void launch_gguf_linear_segment(const __nv_bfloat16* x, const uint8_t* blocks,
                                GgmlType type, __nv_bfloat16* y, int m, int n, int k,
                                size_t row_bytes, int output_stride,
                                float beta, cudaStream_t stream);

void launch_gguf_embedding(int32_t token, const GgufLinearSegment& segment,
                           __nv_bfloat16* out, cudaStream_t stream);
void launch_gguf_embedding_device(const int32_t* token,
                                  const GgufLinearSegment& segment,
                                  __nv_bfloat16* out, cudaStream_t stream);
void launch_gguf_embedding_batch(const int32_t* tokens, int rows,
                                 const GgufLinearSegment& segment,
                                 __nv_bfloat16* out, cudaStream_t stream);

void launch_gguf_dequant(const uint8_t* blocks, GgmlType type,
                         __nv_bfloat16* out, int n, int k,
                         cudaStream_t stream);

}
