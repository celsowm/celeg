#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <celeg/backend/cuda/utils.cuh>

#include <celeg/checkpoint/formats/gguf.hpp>
#include <celeg/detail/model/types.hpp>

namespace celeg {

// Native GGUF block-quantized GEMV. Dequantizes Q4_K / Q6_K super-blocks on the
// fly while accumulating y[activation_row, output_row] = x * W^T. `blocks` is
// the device-resident raw GGUF payload (row-major over GGUF `n`, each row split
// into 256-element super-blocks). `type` selects the block format.
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

// Reference helper for tests and diagnostics; native embedding lookup uses the
// row/batch APIs above and does not materialize the full matrix.
void launch_gguf_dequant(const uint8_t* blocks, GgmlType type,
                         __nv_bfloat16* out, int n, int k,
                         cudaStream_t stream);

} // namespace celeg
