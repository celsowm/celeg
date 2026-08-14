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
// Flash attention prefill kernel.
//
// Tiled online-softmax (flash-attention v1) that keeps Q/K/V in shared memory
// and avoids materializing the full [heads, rows, rows] score matrix. For
// head_dim=64 the cuBLAS batched GEMM path is inefficient (k=64 is too skinny
// for tensor cores, and 16 separate cuBLAS calls add launch overhead). This
// kernel fuses QK^T → softmax → PV in a single kernel launch per attention
// layer, with shared-memory tiling for data reuse.
//
// Tiling: Br=64 query rows per block, Bc=64 KV columns per tile. One block
// per (q_tile, q_head). 256 threads (8 warps), each warp handles 8 rows.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_xor_max(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffffu, val, o));
    return val;
}

__device__ __forceinline__ float warp_xor_sum(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, o);
    return val;
}

__global__ void flash_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    int rows, int q_heads, int kv_heads, int head_dim,
    int q_width, int kv_width, int out_width, int sliding_window) {

    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / 32;
    constexpr int kRowsPerWarp = Br / kWarps;
    constexpr int kMaxAccPerLane = 4;
    // s_k's QK^T read has every lane in a warp read the same `d` at row
    // `col == lane`, i.e. stride head_dim across lanes. For every head_dim
    // celeg supports (64/128) that stride is a multiple of the 32-bank cycle,
    // so all 32 lanes land in the same bank without this pad -- a 32-way
    // serialized conflict on the kernel's hottest loop. s_q's read in the same
    // loop is a broadcast (every lane reads the same address) so it needs no
    // padding, and s_v's read is already lane-contiguous.
    constexpr int kKPad = 8;

    const int q_tile = blockIdx.x;
    const int q_head = blockIdx.y;
    const int group = q_heads / kv_heads;
    const int kv_head = q_head / group;
    const int q_start = q_tile * Br;
    if (q_start >= rows) return;

    const int q_end = min(q_start + Br, rows);
    const int actual_br = q_end - q_start;
    const float scale = rsqrtf((float)head_dim);
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int k_stride = head_dim + kKPad;

    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* s_q = smem;
    __nv_bfloat16* s_k = s_q + Br * head_dim;
    __nv_bfloat16* s_v = s_k + Bc * k_stride;
    float* s_s = reinterpret_cast<float*>(s_v + Bc * head_dim);

    float row_max[kRowsPerWarp];
    float row_denom[kRowsPerWarp];
    float row_acc[kRowsPerWarp][kMaxAccPerLane];

    #pragma unroll
    for (int i = 0; i < kRowsPerWarp; ++i) {
        row_max[i] = -FLT_MAX;
        row_denom[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < kMaxAccPerLane; ++j) row_acc[i][j] = 0.0f;
    }

    const __nv_bfloat16* q_base = q + static_cast<size_t>(q_head) * head_dim;
    for (int i = tid; i < actual_br * head_dim; i += kThreads) {
        int row = i / head_dim;
        int dim = i % head_dim;
        s_q[row * head_dim + dim] = q_base[(q_start + row) * q_width + dim];
    }
    __syncthreads();

    const int num_k_tiles = (q_end + Bc - 1) / Bc;
    const __nv_bfloat16* k_base = k + static_cast<size_t>(kv_head) * head_dim;
    const __nv_bfloat16* v_base = v + static_cast<size_t>(kv_head) * head_dim;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int k_start = kt * Bc;
        int k_end_val = min(k_start + Bc, q_end);
        int actual_bc = k_end_val - k_start;

        for (int i = tid; i < actual_bc * head_dim; i += kThreads) {
            int row = i / head_dim;
            int dim = i % head_dim;
            s_k[row * k_stride + dim] = k_base[(k_start + row) * kv_width + dim];
            s_v[row * head_dim + dim] = v_base[(k_start + row) * kv_width + dim];
        }
        __syncthreads();

        for (int wr = 0; wr < kRowsPerWarp; ++wr) {
            int row = warp * kRowsPerWarp + wr;
            if (row >= actual_br) break;
            for (int col = lane; col < actual_bc; col += 32) {
                const int token = k_start + col;
                const int query_position = q_start + row;
                if (token > query_position ||
                    (sliding_window > 0 && token < query_position + 1 - sliding_window)) {
                    s_s[row * Bc + col] = -FLT_MAX;
                    continue;
                }
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += bf16_float(s_q[row * head_dim + d]) *
                           bf16_float(s_k[col * k_stride + d]);
                }
                s_s[row * Bc + col] = dot * scale;
            }
        }
        __syncthreads();

        for (int wr = 0; wr < kRowsPerWarp; ++wr) {
            int row = warp * kRowsPerWarp + wr;
            if (row >= actual_br) break;

            float tile_max = -FLT_MAX;
            for (int col = lane; col < actual_bc; col += 32) {
                const int token = k_start + col;
                const int query_position = q_start + row;
                if (token <= query_position &&
                    (sliding_window <= 0 || token >= query_position + 1 - sliding_window))
                    tile_max = fmaxf(tile_max, s_s[row * Bc + col]);
            }
            tile_max = warp_xor_max(tile_max);

            float alpha = expf(row_max[wr] - tile_max);
            float new_denom = row_denom[wr] * alpha;

            float tile_sum = 0.0f;
            for (int col = lane; col < actual_bc; col += 32) {
                float p = 0.0f;
                const int token = k_start + col;
                const int query_position = q_start + row;
                if (token <= query_position &&
                    (sliding_window <= 0 || token >= query_position + 1 - sliding_window)) {
                    p = expf(s_s[row * Bc + col] - tile_max);
                    tile_sum += p;
                }
                s_s[row * Bc + col] = p;
            }
            tile_sum = warp_xor_sum(tile_sum);
            new_denom += tile_sum;

            for (int d = lane; d < head_dim; d += 32) {
                int d_idx = d / 32;
                float acc = row_acc[wr][d_idx] * alpha;
                for (int col = 0; col < actual_bc; ++col) {
                    const int token = k_start + col;
                    const int query_position = q_start + row;
                    if (token > query_position) break;
                    if (sliding_window > 0 && token < query_position + 1 - sliding_window)
                        continue;
                    acc += s_s[row * Bc + col] * bf16_float(s_v[col * head_dim + d]);
                }
                row_acc[wr][d_idx] = acc;
            }

            row_max[wr] = tile_max;
            row_denom[wr] = new_denom;
        }
        __syncthreads();
    }

    __nv_bfloat16* out_base = out + static_cast<size_t>(q_head) * head_dim;
    for (int wr = 0; wr < kRowsPerWarp; ++wr) {
        int row = warp * kRowsPerWarp + wr;
        if (row >= actual_br) break;
        for (int d = lane; d < head_dim; d += 32) {
            int d_idx = d / 32;
            float val = row_acc[wr][d_idx] / row_denom[wr];
            out_base[(q_start + row) * out_width + d] = __float2bfloat16(val);
        }
    }
}

void launch_gqa_prefill_flash(const GqaPrefillFlashArgs& args) {
    const int rows = args.rows;
    const int q_heads = args.geometry.q_heads;
    const int head_dim = args.geometry.head_dim;
    const cudaStream_t stream = args.stream;
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kKPad = 8;
    // row_acc[][kMaxAccPerLane] in the kernel gives each lane 4 float slots
    // for its strided (d = lane, lane+32, lane+64, lane+96) accumulation, so
    // head_dim above 128 would index past the array. Every caller currently
    // gates on this externally (head_dim <= 128); assert it here too so a new
    // caller fails loudly instead of silently corrupting registers.
    if (head_dim > 128) {
        throw std::invalid_argument(
            "launch_gqa_prefill_flash: head_dim exceeds the kernel's supported maximum (128)");
    }
    const int num_q_tiles = (rows + Br - 1) / Br;
    dim3 grid(num_q_tiles, q_heads);
    dim3 block(256);
    const size_t smem_bytes =
        (static_cast<size_t>(Br) * head_dim +
         static_cast<size_t>(Bc) * (head_dim + kKPad) +
         static_cast<size_t>(Bc) * head_dim) * sizeof(__nv_bfloat16) +
        static_cast<size_t>(Br) * Bc * sizeof(float);
    // cudaFuncSetAttribute is a driver call, issued once per attention layer
    // per prefill chunk if unconditional -- cheap in isolation, but pure
    // overhead once a given device has already been configured for at least
    // this many bytes. Cache the configured size per device (mirrors
    // mmq_tensor_core_supported()'s per-device cache, for the same reason:
    // callers may legally switch the current CUDA device between calls).
    static std::mutex smem_mutex;
    static std::unordered_map<int, size_t> configured_smem_bytes_by_device;
    int device = 0;
    const bool have_device = cudaGetDevice(&device) == cudaSuccess;
    bool needs_configure = true;
    if (have_device) {
        std::lock_guard<std::mutex> lock(smem_mutex);
        size_t& configured = configured_smem_bytes_by_device[device];
        needs_configure = smem_bytes > configured;
        if (needs_configure) configured = smem_bytes;
    }
    if (needs_configure) {
        CELEG_CUDA(cudaFuncSetAttribute(flash_attn_prefill_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes)));
    }
    flash_attn_prefill_kernel<<<grid, block, smem_bytes, stream>>>(
        args.query, args.kv.keys, args.kv.values, args.out, rows, q_heads,
        args.geometry.kv_heads, head_dim, args.strides.q_width,
        args.strides.kv_width, args.strides.out_width,
        args.geometry.sliding_window);
    CELEG_KERNEL_CHECK();
}

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
                                      int rows, int q_heads, int sliding_window) {
    const int flat = blockIdx.x;
    const int head = flat / rows;
    const int row = flat % rows;
    if (head >= q_heads) return;
    const size_t base = (static_cast<size_t>(head) * rows + row) * static_cast<size_t>(rows);
    const int valid = row + 1;
    const int first = sliding_window > 0 ? max(0, valid - sliding_window) : 0;
    const int lane = threadIdx.x;
    const int threads = blockDim.x;

    __shared__ float warp_scratch[32];
    __shared__ float block_value;

    float local_max = -FLT_MAX;
    for (int c = first + lane; c < valid; c += threads) {
        local_max = fmaxf(local_max, scores[base + c]);
    }
    const float row_max = block_max(local_max, warp_scratch, &block_value);

    float local_sum = 0.0f;
    for (int c = first + lane; c < valid; c += threads) {
        local_sum += expf(scores[base + c] - row_max);
    }
    const float row_sum = block_sum(local_sum, warp_scratch, &block_value);
    const float inv_sum = 1.0f / row_sum;

    for (int c = lane; c < rows; c += threads) {
        const float p = c >= first && c < valid ? expf(scores[base + c] - row_max) * inv_sum : 0.0f;
        probs[base + c] = __float2bfloat16(p);
    }
}

void launch_causal_softmax(const float* scores, __nv_bfloat16* probs,
                           int rows, int q_heads, int sliding_window,
                           cudaStream_t stream) {
    const int threads = rows < 256 ? ((rows + 31) / 32) * 32 : 256;
    causal_softmax_kernel<<<rows * q_heads, max(threads, 32), 0, stream>>>(
        scores, probs, rows, q_heads, sliding_window);
    CELEG_KERNEL_CHECK();
}

void launch_gqa_prefill_gemm(const GqaPrefillGemmArgs& args) {
    const cublasHandle_t cublas = args.cublas;
    const __nv_bfloat16* q = args.query;
    const __nv_bfloat16* k = args.kv.keys;
    const __nv_bfloat16* v = args.kv.values;
    __nv_bfloat16* const out = args.out;
    float* const scores_scratch = args.scores_scratch;
    __nv_bfloat16* const probs_scratch = args.probs_scratch;
    const int rows = args.rows;
    const int q_heads = args.geometry.q_heads;
    const int kv_heads = args.geometry.kv_heads;
    const int head_dim = args.geometry.head_dim;
    const int q_width = args.strides.q_width;
    const int kv_width = args.strides.kv_width;
    const int out_width = args.strides.out_width;
    const int sliding_window = args.geometry.sliding_window;
    const cudaStream_t stream = args.stream;
    const int group = q_heads / kv_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const float zero = 0.0f;
    const float one = 1.0f;
    const long long rows_sq = static_cast<long long>(rows) * rows;

    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* k_base = k + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* q_base = q +
            static_cast<size_t>(kv_head) * group * head_dim;
        float* scores_base = scores_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        CELEG_CUBLAS(cublasGemmStridedBatchedEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            rows, rows, head_dim,
            &scale,
            k_base, CUDA_R_16BF, kv_width, 0,
            q_base, CUDA_R_16BF, q_width, head_dim,
            &zero,
            scores_base, CUDA_R_32F, rows, rows_sq,
            group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    launch_causal_softmax(scores_scratch, probs_scratch, rows, q_heads,
                          sliding_window, stream);

    for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const __nv_bfloat16* v_base = v + static_cast<size_t>(kv_head) * head_dim;
        const __nv_bfloat16* probs_base = probs_scratch +
            static_cast<size_t>(kv_head) * group * rows_sq;
        __nv_bfloat16* out_base = out +
            static_cast<size_t>(kv_head) * group * head_dim;
        CELEG_CUBLAS(cublasGemmStridedBatchedEx(
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
