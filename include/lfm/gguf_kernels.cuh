#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <lfm/gguf.hpp>

namespace lfm {

// Native GGUF block-quantized GEMV. Dequantizes Q4_K / Q6_K super-blocks on the
// fly while accumulating y[activation_row, output_row] = x * W^T. `blocks` is
// the device-resident raw GGUF payload (row-major over GGUF `n`, each row split
// into 256-element super-blocks). `type` selects the block format.
void launch_gguf_linear(const __nv_bfloat16* x, const uint8_t* blocks,
                        GgmlType type, __nv_bfloat16* y, int m, int n, int k,
                        float beta, cudaStream_t stream);

// Dequantizes an entire [n, k] GGUF block-quantized matrix to row-major BF16.
// Used to build the embedding lookup table so token gather reuses the BF16 fast
// path.
void launch_gguf_dequant(const uint8_t* blocks, GgmlType type,
                         __nv_bfloat16* out, int n, int k,
                         cudaStream_t stream);

} // namespace lfm
