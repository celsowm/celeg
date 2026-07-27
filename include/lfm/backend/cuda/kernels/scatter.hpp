#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace lfm {

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

} // namespace lfm
