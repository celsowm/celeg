#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "celeg/backend/cuda/runtime_types.hpp"
#include <cstddef>
#include <cstdint>

namespace celeg {

CudaDeviceCapabilities discover_cuda_device_capabilities();

// Block size for the Q8_1 activation quantization used by the MMQ path
// below (matches ggml's QK8_1).
constexpr int kMmqQ8_1BlockSize = 32;

// Quantizes an activation matrix x[rows,k] (BF16, row-major) into Q8_1:
// one signed int8 per element plus, per 32-element block, a float delta
// (dequant scale) and the block's raw integer sum (sum of the quantized
// int8 values, *not* multiplied by delta - the MMQ kernel needs the raw
// sum to cheaply fold in a Q4_K/Q6_K block's "min" term without a second
// pass over the activations). `k` must be a multiple of kMmqQ8_1BlockSize.
// `scales`/`sums` need rows*(k/32) floats each.
void launch_quantize_q8_1(const __nv_bfloat16* x, int8_t* q8, float* scales,
                          float* sums, int rows, int k, cudaStream_t stream);

// Q4_K x Q8_1 matmul via __dp4a: dequantizes each Q4_K sub-block's scale
// and min once, integer-dot-products its 32 nibbles against the matching
// Q8_1 activation block (8 dp4a calls), and folds in the "min" term via
// the block's precomputed activation sum - no per-element float multiply
// or nibble unpack outside of that. Keeps weights at their native ~4.5
// bits/element on device (no BF16 dequant, unlike the default
// dequant+cuBLAS path from the loader), at the cost of int8 quantization
// noise in the activations. `blocks`/`row_bytes` are the raw GGUF Q4_K
// payload and per-row byte stride, as produced by the GGUF loader/parser.
void launch_q4k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream);

// Q6_K x Q8_1 matmul via __dp4a: same strategy as Q4_K but adapted for the
// Q6_K super-block (6-bit values, per-16-element scales). Each 32-element
// Q8_1 block spans two Q6_K scale sub-blocks, so the dot product is split
// into two 16-element halves with separate scale application. The -32
// weight offset is folded in via the per-16-element activation sum.
void launch_q6k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream);

void launch_q4k_mmq_with_policy(
    const int8_t* q8, const float* q8_scales, const float* q8_sums,
    const uint8_t* blocks, __nv_bfloat16* y, int m, int n, int k,
    size_t row_bytes, int output_stride, float beta, bool use_tensor_cores,
    cudaStream_t stream);
void launch_q6k_mmq_with_policy(
    const int8_t* q8, const float* q8_scales, const float* q8_sums,
    const uint8_t* blocks, __nv_bfloat16* y, int m, int n, int k,
    size_t row_bytes, int output_stride, float beta, bool use_tensor_cores,
    cudaStream_t stream);

} // namespace celeg
