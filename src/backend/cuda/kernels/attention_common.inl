// Device helpers shared by every attention strategy in this directory.
//
// Two reduction styles coexist deliberately:
//   * attention_dot / attention_dot_int8 reduce across the whole block via
//     block_sum (two __syncthreads()), used by the strict reference kernels
//     where every thread must observe the same score before proceeding.
//   * warp_broadcast_sum reduces within a single warp via shuffles only, used
//     by the online/segmented kernels whose blocks are exactly one warp.
//
// These are plain __device__ (external linkage), which is safe only because
// every attention_*.inl leaf is textually included into the single attention.cu
// translation unit. Do not promote a leaf to its own .cu without first giving
// these internal linkage.

__device__ float attention_dot_int8(const __nv_bfloat16* query,
                                    const int8_t* key, float key_scale,
                                    int head_dim, float* warp_sums,
                                    float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) *
            (static_cast<float>(key[d]) * key_scale);
    }
    return block_sum(partial, warp_sums, total);
}

__device__ float attention_dot(const __nv_bfloat16* query,
                               const __nv_bfloat16* key,
                               int head_dim,
                               float* warp_sums,
                               float* total) {
    float partial = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        partial += bf16_float(query[d]) * bf16_float(key[d]);
    }
    return block_sum(partial, warp_sums, total);
}

// Maximum head_dim this file's warp-only decode-attention kernels support
// (32 lanes * kMaxHeadDimPerLane each); LFM2/LFM2.5 head dims (64-128) are
// well within this.
constexpr int kMaxHeadDimPerLane = 8;

// One warp handles one (query row, query head) pair for the whole KV loop,
// with no block-wide synchronization at all: the Q.K dot product is reduced
// with warp_sum + a __shfl_sync broadcast (every lane ends up with the same
// scalar), so every lane can independently recompute the online-softmax
// running max/denominator from that broadcast value and keep its own slice
// of the V-weighted accumulator in registers. Contrast with the block-wide
// design below (attention_dot + block_sum), which pays two __syncthreads()
// per KV token; for a 512-token context that is over a thousand block-wide
// barriers per decode step, serializing every warp in the block against the
// slowest one on every single token.
__device__ __forceinline__ float warp_broadcast_sum(float partial) {
    return __shfl_sync(0xffffffffu, warp_sum(partial), 0);
}
