#include "kernel_common.cuh"
#include "celeg/model/runtime_types.hpp"

namespace celeg {

__global__ void argmax_bf16_kernel(const __nv_bfloat16* values,
                                   const std::uint8_t* seen,
                                   int count, float repetition_penalty,
                                   int32_t* result) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    float best = -FLT_MAX;
    int index = -1;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float value = bf16_float(values[i]);
        if (seen[i]) {
            value = value < 0.0f ? value * repetition_penalty
                                 : value / repetition_penalty;
        }
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
// whole vocabulary. Benchmark measurements show that one loop was
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
// `index_base` is added to the emitted index only (never to the comparisons
// or the refill-ownership arithmetic), so a caller reducing a slice
// `scores + base` of a larger array can recover absolute indices without
// changing any of the ordering/refill logic below.
__device__ void block_select_topk(const float* scores, int vocab, int ranks,
                                  float* out_values, int32_t* out_indices,
                                  float* best_values, int* best_indices,
                                  float* reduce_values, int* reduce_indices,
                                  int index_base = 0) {
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
            out_indices[rank] = won_index == INT_MAX ? INT_MAX : won_index + index_base;
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

// Stage 1 of the grid-parallel top-k path: each block scores and reduces one
// contiguous slice of the vocabulary [start, end) to its own local top
// `ranks` candidates (same temperature/repetition-penalty adjustment and the
// same block_select_topk ordering as the single-block path), writing them to
// partial_values/partial_indices[blockIdx.x * ranks .. +ranks). Splitting the
// O(vocab/blockDim.x) scan across gridDim.x blocks spreads it across that
// many SMs instead of just one, which matters because the single-block path
// only ever occupies one SM regardless of how wide it is.
__global__ void sample_topk_partial_kernel(const __nv_bfloat16* logits,
                                           const uint8_t* seen,
                                           float* scores,
                                           int vocab,
                                           float temperature,
                                           float repetition_penalty,
                                           bool greedy,
                                           int ranks,
                                           float* partial_values,
                                           int32_t* partial_indices) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    __shared__ float reduce_values[256];
    __shared__ int reduce_indices[256];

    const int num_partitions = gridDim.x;
    const int partition = blockIdx.x;
    const int start = static_cast<int>((static_cast<int64_t>(partition) * vocab) / num_partitions);
    const int end = static_cast<int>((static_cast<int64_t>(partition + 1) * vocab) / num_partitions);
    const int count = end - start;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int index = start + i;
        float value = bf16_float(logits[index]);
        if (seen[index]) {
            value = value < 0.0f ? value * repetition_penalty : value / repetition_penalty;
        }
        scores[index] = greedy ? value : value / temperature;
    }
    __syncthreads();

    block_select_topk(scores + start, count, ranks,
                      partial_values + static_cast<size_t>(partition) * ranks,
                      partial_indices + static_cast<size_t>(partition) * ranks,
                      best_values, best_indices, reduce_values, reduce_indices, start);
}

// Stage 2: merges the gridDim.x(stage1) * ranks partial candidates (a couple
// hundred to low thousands of entries, an order of magnitude smaller than
// vocab) into the final ordered top-k and samples from it. Candidate values
// and vocabulary indices live in separate arrays here (unlike
// block_select_topk, where index == array position), so this does its own
// small-scale destructive block-parallel argmax over a shared-memory copy
// instead of reusing block_select_topk directly. Vocabulary indices are
// globally unique across stage-1 partitions (they cover disjoint vocab
// ranges), so removing a winner by index match never removes more than one
// candidate.
__global__ void sample_topk_merge_kernel(const float* partial_values,
                                         const int32_t* partial_indices,
                                         int candidate_count,
                                         int ranks,
                                         bool greedy,
                                         float top_p,
                                         uint64_t* rng_state,
                                         uint8_t* seen,
                                         int vocab,
                                         float* selected_values,
                                         int32_t* selected_indices,
                                         int32_t* result) {
    extern __shared__ char merge_smem[];
    float* candidate_values = reinterpret_cast<float*>(merge_smem);
    int32_t* candidate_indices = reinterpret_cast<int32_t*>(candidate_values + candidate_count);
    __shared__ float best_values[256];
    __shared__ int best_indices[256];

    for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
        candidate_values[j] = partial_values[j];
        candidate_indices[j] = partial_indices[j];
    }
    __syncthreads();

    for (int rank = 0; rank < ranks; ++rank) {
        float bv = -FLT_MAX;
        int bi = INT_MAX;
        for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
            const int32_t vi = candidate_indices[j];
            if (vi == INT_MAX) continue;
            if (sampling_candidate_better(candidate_values[j], vi, bv, bi)) {
                bv = candidate_values[j];
                bi = vi;
            }
        }
        best_values[threadIdx.x] = bv;
        best_indices[threadIdx.x] = bi;
        __syncthreads();
        for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
            if (threadIdx.x < step &&
                sampling_candidate_better(best_values[threadIdx.x + step], best_indices[threadIdx.x + step],
                                          best_values[threadIdx.x], best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = best_values[threadIdx.x + step];
                best_indices[threadIdx.x] = best_indices[threadIdx.x + step];
            }
            __syncthreads();
        }
        const float won_value = best_values[0];
        const int won_index = best_indices[0];
        if (threadIdx.x == 0) {
            selected_values[rank] = won_value;
            selected_indices[rank] = won_index;
        }
        for (int j = threadIdx.x; j < candidate_count; j += blockDim.x) {
            if (won_index != INT_MAX && candidate_indices[j] == won_index) {
                candidate_indices[j] = INT_MAX;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int chosen_token = selected_indices[0];
        if (!greedy) {
            const float maximum = selected_values[0];
            float total = 0.0f;
            for (int i = 0; i < ranks; ++i) total += expf(selected_values[i] - maximum);

            int cutoff = ranks;
            if (top_p < 1.0f) {
                float cumulative = 0.0f;
                for (int i = 0; i < ranks; ++i) {
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
            chosen_token = selected_indices[chosen];
        }
        *result = chosen_token;
        if (chosen_token >= 0 && chosen_token < vocab) seen[chosen_token] = 1;
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
    int selected_stride,
    int32_t* result) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* row_logits = logits[row];
    uint8_t* row_seen = seen[row];
    float* row_scores = scores + static_cast<size_t>(row) * vocab;
    float* row_selected_values =
        selected_values + static_cast<size_t>(row) * selected_stride;
    int32_t* row_selected_indices =
        selected_indices + static_cast<size_t>(row) * selected_stride;
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

void launch_argmax_bf16(const __nv_bfloat16* logits, const std::uint8_t* seen,
                        int count, float repetition_penalty, int32_t* result,
                        cudaStream_t stream) {
    argmax_bf16_kernel<<<1, 256, 0, stream>>>(
        logits, seen, count, repetition_penalty, result);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch(const int32_t* tokens, int count,
                            uint8_t* seen, int vocab, cudaStream_t stream) {
    mark_seen_batch_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        tokens, count, seen, vocab);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen_batch_ptrs(const int32_t* tokens,
                                 uint8_t* const* seen,
                                 int rows, int vocab,
                                 cudaStream_t stream) {
    mark_seen_batch_ptrs_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(
        tokens, seen, rows, vocab);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_mark_seen(const int32_t* token, uint8_t* seen, int vocab,
                      cudaStream_t stream) {
    mark_seen_kernel<<<1, 1, 0, stream>>>(token, seen, vocab);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_prepare_sampling_scores(const __nv_bfloat16* logits,
                                    const uint8_t* seen,
                                    float* scores, int vocab,
                                    float temperature,
                                    float repetition_penalty,
                                    cudaStream_t stream) {
    prepare_sampling_scores_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
        logits, seen, scores, vocab, temperature, repetition_penalty);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_select_topk(float* scores, float* selected_values,
                        int32_t* selected_indices, int rank, int vocab,
                        cudaStream_t stream) {
    select_topk_kernel<<<1, 256, 0, stream>>>(
        scores, selected_values, selected_indices, rank, vocab);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_sample_topk(const float* selected_values,
                        const int32_t* selected_indices,
                        int top_k, float top_p, uint64_t* rng_state,
                        int32_t* result, cudaStream_t stream) {
    sample_topk_kernel<<<1, 1, 0, stream>>>(
        selected_values, selected_indices, top_k, top_p, rng_state, result);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

void launch_fused_sample_topk(const __nv_bfloat16* logits,
                              uint8_t* seen,
                              float* scores,
                              float* selected_values,
                              int32_t* selected_indices,
                              float* partial_values,
                              int32_t* partial_indices,
                              int vocab,
                              float temperature,
                              float repetition_penalty,
                              int top_k,
                              float top_p,
                              uint64_t* rng_state,
                              int32_t* result,
                              cudaStream_t stream) {
    // Below the threshold a single 256-thread block already finishes the
    // scan quickly and a second kernel launch would only add overhead, so
    // only split large vocabularies (where kSamplingPartialBlocks actually
    // buys meaningfully more parallelism than one block) across the grid.
    constexpr int kPartialThreshold = kSamplingPartialBlocks * 256 * 4;
    if (vocab >= kPartialThreshold) {
        const int candidate_count = kSamplingPartialBlocks * top_k;
        sample_topk_partial_kernel<<<kSamplingPartialBlocks, 256, 0, stream>>>(
            logits, seen, scores, vocab, temperature, repetition_penalty,
            /*greedy=*/false, top_k, partial_values, partial_indices);
        const size_t merge_smem = static_cast<size_t>(candidate_count) *
                                  (sizeof(float) + sizeof(int32_t));
        sample_topk_merge_kernel<<<1, 256, merge_smem, stream>>>(
            partial_values, partial_indices, candidate_count, top_k,
            /*greedy=*/false, top_p, rng_state, seen, vocab,
            selected_values, selected_indices, result);
    } else {
        fused_sample_topk_kernel<<<1, 256, 0, stream>>>(
            logits, seen, scores, selected_values, selected_indices, vocab,
            temperature, repetition_penalty, top_k, top_p, rng_state, result);
    }
    CELEG_KERNEL_DEBUG_SYNC(stream);
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
    int selected_stride,
    int32_t* result,
    cudaStream_t stream) {
    packed_sample_topk_kernel<<<rows, 256, 0, stream>>>(
        logits, seen, rng_state, temperatures, repetition_penalties,
        top_k, top_p, scores, selected_values, selected_indices,
        rows, vocab, selected_stride, result);
    CELEG_KERNEL_DEBUG_SYNC(stream);
}

} // namespace celeg
