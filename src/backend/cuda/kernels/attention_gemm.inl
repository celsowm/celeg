// Causal prefill attention expressed as batched GEMM instead of hand-written
// kernels: cuBLAS computes QK^T and PV via cublasGemmStridedBatchedEx (with
// stride 0 on the KV operand to broadcast across a GQA group), with a parallel
// causal softmax kernel in between.
//
// This is the fast prefill path -- it reaches tensor cores, which the scalar
// per-token kernels in the other leaves cannot. It costs O(rows^2) scratch for
// the score/probability matrices, so callers fall back to
// launch_gqa_prefill_segmented above a row-count ceiling.
//
// The only leaf that touches cuBLAS.

// ---------------------------------------------------------------------------
// Batched-GEMM causal prefill attention.
//
// The per-(row,head) kernels above compute one scalar dot product per KV
// token per thread-block, with no tensor cores - fine for decode (one query
// against a long KV history) but the wrong shape for prefill, where every
// row attends to a *different* prefix of the *same* rows. That is exactly
// the shape cuBLAS batched GEMM wants: for each head, QK^T and softmax(.)V
// are dense matrix multiplies. This path computes raw scores with one
// strided-batched GEMM per KV-head group (broadcasting the shared K/V
// across the q_heads/kv_heads queries in that GQA group via strideB/strideA
// = 0), a causal softmax kernel, then a second strided-batched GEMM for the
// value contraction. Two cuBLAS calls per KV head instead of one scalar
// dot product per KV token per row.
// ---------------------------------------------------------------------------

// One block per (head, row). Reduces over the causal prefix [0, row] twice
// (max, then sum of exp) using the existing block_max/block_sum reductions,
// then writes a full-width BF16 probability row (zero past `row`) so the
// following dense PV GEMM naturally ignores masked positions.
__global__ void causal_softmax_kernel(const float* __restrict__ scores,
                                      __nv_bfloat16* __restrict__ probs,
                                      int rows, int q_heads) {
    const int flat = blockIdx.x;
    const int head = flat / rows;
    const int row = flat % rows;
    if (head >= q_heads) return;
    const size_t base = (static_cast<size_t>(head) * rows + row) * static_cast<size_t>(rows);
    const int valid = row + 1;
    const int lane = threadIdx.x;
    const int threads = blockDim.x;

    __shared__ float warp_scratch[32];
    __shared__ float block_value;

    float local_max = -FLT_MAX;
    for (int c = lane; c < valid; c += threads) {
        local_max = fmaxf(local_max, scores[base + c]);
    }
    const float row_max = block_max(local_max, warp_scratch, &block_value);

    float local_sum = 0.0f;
    for (int c = lane; c < valid; c += threads) {
        local_sum += expf(scores[base + c] - row_max);
    }
    const float row_sum = block_sum(local_sum, warp_scratch, &block_value);
    const float inv_sum = 1.0f / row_sum;

    for (int c = lane; c < rows; c += threads) {
        const float p = c < valid ? expf(scores[base + c] - row_max) * inv_sum : 0.0f;
        probs[base + c] = __float2bfloat16(p);
    }
}

void launch_causal_softmax(const float* scores, __nv_bfloat16* probs,
                           int rows, int q_heads, cudaStream_t stream) {
    const int threads = rows < 256 ? ((rows + 31) / 32) * 32 : 256;
    causal_softmax_kernel<<<rows * q_heads, max(threads, 32), 0, stream>>>(
        scores, probs, rows, q_heads);
    LFM_KERNEL_CHECK();
}

void launch_gqa_prefill_gemm(
    cublasHandle_t cublas, const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* out, float* scores_scratch,
    __nv_bfloat16* probs_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int q_width, int kv_width, int out_width,
    cudaStream_t stream) {
    const int group = q_heads / kv_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float zero = 0.0f;
    const long long rows_sq = static_cast<long long>(rows) * rows;

    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* k_base = k + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* q_base = q +
            static_cast<size_t>(kv_head) * group * head_dim;
        float* scores_base = scores_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        // scores[rows,rows] (row-major) = Q[rows,head_dim] @ K^T[head_dim,rows],
        // scaled by 1/sqrt(head_dim). Same col-major transpose recipe as
        // GemmDispatcher::linear_cublas, batched over the `group` queries
        // that share this KV head (K is broadcast: strideA = 0).
        LFM_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, rows, head_dim,
            &scale,
            k_base, CUDA_R_16BF, kv_width, 0,
            q_base, CUDA_R_16BF, q_width, head_dim,
            &zero,
            scores_base, CUDA_R_32F, rows, rows_sq,
            group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    launch_causal_softmax(scores_scratch, probs_scratch, rows, q_heads, stream);

    const float one = 1.0f;
    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* v_base = v + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* probs_base = probs_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        __nv_bfloat16* out_base = out +
            static_cast<size_t>(kv_head) * group * head_dim;
        // out[rows,head_dim] (row-major) = P[rows,rows] @ V[rows,head_dim].
        // Row-major-via-col-major no-transpose recipe, batched over the
        // group's queries (V is broadcast: strideA = 0 in this call's A
        // role, which is our math "B"/V).
        LFM_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            head_dim, rows, rows,
            &one,
            v_base, CUDA_R_16BF, kv_width, 0,
            probs_base, CUDA_R_16BF, rows, rows_sq,
            &zero,
            out_base, CUDA_R_16BF, out_width, head_dim,
            group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}
