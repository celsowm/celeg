#include "kernel_common.cuh"

namespace lfm {

__global__ void argmax_bf16_kernel(const __nv_bfloat16* values,
                                   int count,
                                   int32_t* result) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    float best = -FLT_MAX;
    int index = -1;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const float value = bf16_float(values[i]);
        if (value > best || (value == best && (index < 0 || i < index))) {
            best = value;
            index = i;
        }
    }

    best_values[threadIdx.x] = best;
    best_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const float other_value = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other_value > best_values[threadIdx.x] ||
                (other_value == best_values[threadIdx.x] && other_index >= 0 &&
                 (best_indices[threadIdx.x] < 0 || other_index < best_indices[threadIdx.x]))) {
                best_values[threadIdx.x] = other_value;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *result = best_indices[0];
}

__global__ void mark_seen_batch_kernel(const int32_t* tokens,
                                       int count,
                                       uint8_t* seen,
                                       int vocab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const int token = tokens[i];
        if (token >= 0 && token < vocab) seen[token] = 1;
    }
}

__global__ void mark_seen_batch_ptrs_kernel(const int32_t* tokens,
                                            uint8_t* const* seen,
                                            int rows,
                                            int vocab) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int token = tokens[row];
    if (token >= 0 && token < vocab) seen[row][token] = 1;
}

__global__ void mark_seen_kernel(const int32_t* token,
                                 uint8_t* seen,
                                 int vocab) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const int value = *token;
        if (value >= 0 && value < vocab) seen[value] = 1;
    }
}

__global__ void prepare_sampling_scores_kernel(const __nv_bfloat16* logits,
                                               const uint8_t* seen,
                                               float* scores,
                                               int vocab,
                                               float temperature,
                                               float repetition_penalty) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    float value = bf16_float(logits[i]);
    if (seen[i]) {
        value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
    }
    scores[i] = value / temperature;
}

__global__ void select_topk_kernel(float* scores,
                                   float* selected_values,
                                   int32_t* selected_indices,
                                   int rank,
                                   int vocab) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    float best = -FLT_MAX;
    int index = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float value = scores[i];
        if (value > best || (value == best && (index < 0 || i < index))) {
            best = value;
            index = i;
        }
    }
    best_values[threadIdx.x] = best;
    best_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            const float other_value = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other_value > best_values[threadIdx.x] ||
                (other_value == best_values[threadIdx.x] && other_index >= 0 &&
                 (best_indices[threadIdx.x] < 0 || other_index < best_indices[threadIdx.x]))) {
                best_values[threadIdx.x] = other_value;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        selected_values[rank] = best_values[0];
        selected_indices[rank] = best_indices[0];
        if (best_indices[0] >= 0) scores[best_indices[0]] = -FLT_MAX;
    }
}

__device__ __forceinline__ uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

__global__ void sample_topk_kernel(const float* selected_values,
                                   const int32_t* selected_indices,
                                   int top_k,
                                   float top_p,
                                   uint64_t* rng_state,
                                   int32_t* result) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const float maximum = selected_values[0];
    float total = 0.0f;
    for (int i = 0; i < top_k; ++i) total += expf(selected_values[i] - maximum);

    int cutoff = top_k;
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cumulative += expf(selected_values[i] - maximum);
            if (cumulative / total >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
    }

    float truncated_total = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
        truncated_total += expf(selected_values[i] - maximum);
    }
    uint64_t state = *rng_state;
    const uint64_t random_bits = xorshift64star(state);
    *rng_state = state;
    const double unit = (static_cast<double>(random_bits >> 11) + 0.5) *
                        (1.0 / 9007199254740992.0);
    const float target = static_cast<float>(unit) * truncated_total;
    float cumulative = 0.0f;
    int chosen = cutoff - 1;
    for (int i = 0; i < cutoff; ++i) {
        cumulative += expf(selected_values[i] - maximum);
        if (target <= cumulative) {
            chosen = i;
            break;
        }
    }
    *result = selected_indices[chosen];
}

// Candidate ordering shared by the top-k selection below: higher score wins,
// and on an exact tie the lower vocabulary index wins. This is the ordering the
// original single-threaded insertion sort produced, and callers (and the
// determinism tests) depend on it, so any parallel selection must reproduce it
// exactly rather than merely picking "a" maximum.
__device__ __forceinline__ bool sampling_candidate_better(float va, int ia,
                                                          float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Block-parallel ordered top-k selection over scores[0, vocab).
//
// This replaced a single-threaded (`threadIdx.x == 0`) insertion sort over the
// whole vocabulary. Measured on LFM2.5-230M (vocab 65536) that one loop was
// 5.47 ms of an 8.34 ms decode step -- 66% of decode, more than every GEMM in
// the model combined -- because one lane walked 65536 dependent global loads
// with no parallelism to hide the latency.
//
// Strategy: every thread reduces its own strided slice to a single best
// candidate, then each rank is drained by a block-wide argmax over those
// blockDim.x candidates. Only the thread that owned the winner has to refill,
// and it rescans just its own slice (vocab/blockDim.x elements) for its
// next-best strictly below what was just emitted. So the serial work per rank
// is one slice, not the whole vocabulary.
//
// Ordering is descending score, ties broken by *lower* vocabulary index --
// identical to what the insertion sort produced. Callers depend on this, so it
// is asserted directly in cuda_kernels_test.cu rather than left implicit.
//
// `best_*` / `reduce_*` are caller-provided shared scratch, blockDim.x wide.
// blockDim.x must be a power of two (the reduction halves it).
__device__ void block_select_topk(const float* scores, int vocab, int ranks,
                                  float* out_values, int32_t* out_indices,
                                  float* best_values, int* best_indices,
                                  float* reduce_values, int* reduce_indices) {
    const int stride = blockDim.x;
    {
        float bv = -FLT_MAX;
        int bi = INT_MAX;
        for (int i = threadIdx.x; i < vocab; i += stride) {
            if (sampling_candidate_better(scores[i], i, bv, bi)) { bv = scores[i]; bi = i; }
        }
        best_values[threadIdx.x] = bv;
        best_indices[threadIdx.x] = bi;
    }
    __syncthreads();

    for (int rank = 0; rank < ranks; ++rank) {
        reduce_values[threadIdx.x] = best_values[threadIdx.x];
        reduce_indices[threadIdx.x] = best_indices[threadIdx.x];
        __syncthreads();
        for (int step = stride >> 1; step > 0; step >>= 1) {
            if (threadIdx.x < step &&
                sampling_candidate_better(reduce_values[threadIdx.x + step],
                                          reduce_indices[threadIdx.x + step],
                                          reduce_values[threadIdx.x],
                                          reduce_indices[threadIdx.x])) {
                reduce_values[threadIdx.x] = reduce_values[threadIdx.x + step];
                reduce_indices[threadIdx.x] = reduce_indices[threadIdx.x + step];
            }
            __syncthreads();
        }
        const float won_value = reduce_values[0];
        const int won_index = reduce_indices[0];
        if (threadIdx.x == 0) {
            out_values[rank] = won_value;
            out_indices[rank] = won_index;
        }
        // Refill only the owning thread; thread t owns vocabulary indices
        // t, t+stride, ..., so ownership is recoverable from the index alone.
        if (rank + 1 < ranks && won_index != INT_MAX &&
            threadIdx.x == won_index % stride) {
            float nv = -FLT_MAX;
            int ni = INT_MAX;
            for (int i = threadIdx.x; i < vocab; i += stride) {
                if (sampling_candidate_better(scores[i], i, nv, ni) &&
                    sampling_candidate_better(won_value, won_index, scores[i], i)) {
                    nv = scores[i];
                    ni = i;
                }
            }
            best_values[threadIdx.x] = nv;
            best_indices[threadIdx.x] = ni;
        }
        __syncthreads();
    }
}

__global__ void fused_sample_topk_kernel(const __nv_bfloat16* logits,
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
                                                int32_t* result) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    __shared__ float reduce_values[256];
    __shared__ int reduce_indices[256];

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float value = bf16_float(logits[i]);
        if (seen[i]) {
            value = value < 0.0f ? value * repetition_penalty
                                 : value / repetition_penalty;
        }
        scores[i] = value / temperature;
    }
    __syncthreads();

    block_select_topk(scores, vocab, top_k, selected_values, selected_indices,
                      best_values, best_indices, reduce_values, reduce_indices);

    if (threadIdx.x == 0) {
        const float maximum = selected_values[0];
        float total = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            total += expf(selected_values[i] - maximum);
        }

        int cutoff = top_k;
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (int i = 0; i < top_k; ++i) {
                cumulative += expf(selected_values[i] - maximum);
                if (cumulative / total >= top_p) {
                    cutoff = i + 1;
                    break;
                }
            }
        }

        float truncated_total = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            truncated_total += expf(selected_values[i] - maximum);
        }
        uint64_t state = *rng_state;
        const uint64_t random_bits = xorshift64star(state);
        *rng_state = state;
        const double unit = (static_cast<double>(random_bits >> 11) + 0.5) *
                            (1.0 / 9007199254740992.0);
        const float target = static_cast<float>(unit) * truncated_total;
        float cumulative = 0.0f;
        int chosen = cutoff - 1;
        for (int i = 0; i < cutoff; ++i) {
            cumulative += expf(selected_values[i] - maximum);
            if (target <= cumulative) {
                chosen = i;
                break;
            }
        }
        *result = selected_indices[chosen];
        const int32_t token = *result;
        if (token >= 0 && token < vocab) seen[token] = 1;
    }
}


__global__ void packed_sample_topk_kernel(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k_values,
    const float* top_p_values,
    float* scores,
    float* selected_values,
    int32_t* selected_indices,
    int rows,
    int vocab,
    int32_t* result) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_logits = logits[row];
    uint8_t* row_seen = seen[row];
    float* row_scores = scores + static_cast<size_t>(row) * vocab;
    float* row_selected_values =
        selected_values + static_cast<size_t>(row) * 128;
    int32_t* row_selected_indices =
        selected_indices + static_cast<size_t>(row) * 128;
    const float temperature = temperatures[row];
    const float repetition_penalty = repetition_penalties[row];
    const int top_k = top_k_values[row];
    const float top_p = top_p_values[row];
    const bool greedy = temperature <= 0.0f || top_k == 1;

    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    __shared__ float reduce_values[256];
    __shared__ int reduce_indices[256];

    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float value = bf16_float(row_logits[i]);
        if (row_seen[i]) {
            value = value < 0.0f ? value * repetition_penalty
                                 : value / repetition_penalty;
        }
        row_scores[i] = greedy ? value : value / temperature;
    }
    __syncthreads();

    block_select_topk(row_scores, vocab, greedy ? 1 : top_k,
                      row_selected_values, row_selected_indices,
                      best_values, best_indices, reduce_values, reduce_indices);
    __syncthreads();

    if (threadIdx.x == 0) {
        int chosen_token = row_selected_indices[0];
        if (!greedy) {
            const float maximum = row_selected_values[0];
            float total = 0.0f;
            for (int i = 0; i < top_k; ++i) {
                total += expf(row_selected_values[i] - maximum);
            }
            int cutoff = top_k;
            if (top_p < 1.0f) {
                float cumulative = 0.0f;
                for (int i = 0; i < top_k; ++i) {
                    cumulative += expf(row_selected_values[i] - maximum);
                    if (cumulative / total >= top_p) {
                        cutoff = i + 1;
                        break;
                    }
                }
            }
            float truncated_total = 0.0f;
            for (int i = 0; i < cutoff; ++i) {
                truncated_total += expf(row_selected_values[i] - maximum);
            }
            uint64_t state = *rng_state[row];
            const uint64_t random_bits = xorshift64star(state);
            *rng_state[row] = state;
            const double unit =
                (static_cast<double>(random_bits >> 11) + 0.5) *
                (1.0 / 9007199254740992.0);
            const float target = static_cast<float>(unit) * truncated_total;
            float cumulative = 0.0f;
            int chosen = cutoff - 1;
            for (int i = 0; i < cutoff; ++i) {
                cumulative += expf(row_selected_values[i] - maximum);
                if (target <= cumulative) {
                    chosen = i;
                    break;
                }
            }
            chosen_token = row_selected_indices[chosen];
        }
        result[row] = chosen_token;
        if (chosen_token >= 0 && chosen_token < vocab) {
            row_seen[chosen_token] = 1;
        }
    }
}

void launch_argmax_bf16(const __nv_bfloat16* logits, int count,
                        int32_t* result, cudaStream_t stream) {
    argmax_bf16_kernel<<<1, 256, 0, stream>>>(logits, count, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch(const int32_t* tokens, int count,
                            uint8_t* seen, int vocab, cudaStream_t stream) {
    mark_seen_batch_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        tokens, count, seen, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch_ptrs(const int32_t* tokens,
                                 uint8_t* const* seen,
                                 int rows, int vocab,
                                 cudaStream_t stream) {
    mark_seen_batch_ptrs_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        tokens, seen, rows, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen(const int32_t* token, uint8_t* seen, int vocab,
                      cudaStream_t stream) {
    mark_seen_kernel<<<1, 1, 0, stream>>>(token, seen, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_prepare_sampling_scores(const __nv_bfloat16* logits,
                                    const uint8_t* seen,
                                    float* scores, int vocab,
                                    float temperature,
                                    float repetition_penalty,
                                    cudaStream_t stream) {
    prepare_sampling_scores_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
        logits, seen, scores, vocab, temperature, repetition_penalty);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_select_topk(float* scores, float* selected_values,
                        int32_t* selected_indices, int rank, int vocab,
                        cudaStream_t stream) {
    select_topk_kernel<<<1, 256, 0, stream>>>(
        scores, selected_values, selected_indices, rank, vocab);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

void launch_sample_topk(const float* selected_values,
                        const int32_t* selected_indices,
                        int top_k, float top_p, uint64_t* rng_state,
                        int32_t* result, cudaStream_t stream) {
    sample_topk_kernel<<<1, 1, 0, stream>>>(
        selected_values, selected_indices, top_k, top_p, rng_state, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

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
                              cudaStream_t stream) {
    fused_sample_topk_kernel<<<1, 256, 0, stream>>>(
        logits, seen, scores, selected_values, selected_indices, vocab,
        temperature, repetition_penalty, top_k, top_p, rng_state, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}


void launch_packed_sample_topk(
    __nv_bfloat16* const* logits,
    uint8_t* const* seen,
    uint64_t* const* rng_state,
    const float* temperatures,
    const float* repetition_penalties,
    const int32_t* top_k,
    const float* top_p,
    float* scores,
    float* selected_values,
    int32_t* selected_indices,
    int rows,
    int vocab,
    int32_t* result,
    cudaStream_t stream) {
    packed_sample_topk_kernel<<<rows, 256, 0, stream>>>(
        logits, seen, rng_state, temperatures, repetition_penalties,
        top_k, top_p, scores, selected_values, selected_indices,
        rows, vocab, result);
    LFM_KERNEL_DEBUG_SYNC(stream);
}

} // namespace lfm
