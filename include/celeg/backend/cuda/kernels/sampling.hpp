#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace celeg {

void launch_argmax_bf16(const __nv_bfloat16* logits, int count,
                        int32_t* result, cudaStream_t stream);
void launch_mark_seen_batch(const int32_t* tokens, int count,
                            uint8_t* seen, int vocab, cudaStream_t stream);
void launch_mark_seen_batch_ptrs(const int32_t* tokens,
                                 uint8_t* const* seen,
                                 int rows, int vocab,
                                 cudaStream_t stream);
void launch_mark_seen(const int32_t* token, uint8_t* seen, int vocab,
                      cudaStream_t stream);
void launch_prepare_sampling_scores(const __nv_bfloat16* logits,
                                    const uint8_t* seen,
                                    float* scores, int vocab,
                                    float temperature,
                                    float repetition_penalty,
                                    cudaStream_t stream);
void launch_select_topk(float* scores, float* selected_values,
                        int32_t* selected_indices, int rank, int vocab,
                        cudaStream_t stream);
void launch_sample_topk(const float* selected_values,
                        const int32_t* selected_indices,
                        int top_k, float top_p, uint64_t* rng_state,
                        int32_t* result, cudaStream_t stream);
void launch_fused_sample_topk(const __nv_bfloat16* logits,
                              uint8_t* seen,
                              float* scores,
                              float* selected_values,
                              int32_t* selected_indices,
                              int vocab,
                              float temperature,
                              float repetition_penalty,
                              int top_k,
                              float top_p,
                              uint64_t* rng_state,
                              int32_t* result,
                              cudaStream_t stream);

// Packed multi-session decode helpers. Pointer arrays contain one persistent
// session allocation per packed row; activations stay contiguous by batch.
void launch_packed_sample_topk(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k,
    const float* top_p,
    float* scores, float* selected_values,
    int32_t* selected_indices,
    int rows, int vocab, int32_t* result,
    cudaStream_t stream);

} // namespace celeg
