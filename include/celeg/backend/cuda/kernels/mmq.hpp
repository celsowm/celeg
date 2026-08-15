#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "celeg/backend/cuda/runtime_types.hpp"
#include <cstddef>
#include <cstdint>

namespace celeg {

CudaDeviceCapabilities discover_cuda_device_capabilities();

constexpr int kMmqQ8_1BlockSize = 32;

void launch_quantize_q8_1(const __nv_bfloat16* x, int8_t* q8, float* scales,
                          float* sums, int rows, int k, cudaStream_t stream);

void launch_q4k_mmq(const int8_t* q8, const float* q8_scales,
                    const float* q8_sums, const uint8_t* blocks,
                    __nv_bfloat16* y, int m, int n, int k, size_t row_bytes,
                    int output_stride, float beta, cudaStream_t stream);

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

}
