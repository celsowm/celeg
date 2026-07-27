# Static Audit Triage for `lfm25-cuda-cpp`

Reference points:
- Current tree: `89aa918`
- Historical CUDA implementation used by the pasted audit: `6011b79`

Scope:
- Static validation only
- No source edits
- No profiler traces, benchmark runs, or model-level performance measurements

## Findings

### 1. Source manifests on `89aa918` are broken
**Verdict:** `Verified on current tree`

The audit's top-level buildability concern is correct, but incomplete.

- [cmake/sources/cuda_backend.cmake](/D:/ia/lfm25-cuda-cpp/cmake/sources/cuda_backend.cmake:1) references five missing CUDA model files under `src/backend/cuda/model/`.
- [cmake/sources/base_runtime.cmake](/D:/ia/lfm25-cuda-cpp/cmake/sources/base_runtime.cmake:2) also references a missing `src/model/...` subtree.
- `6011b79` still had the old flat CUDA files (`src/model.cu`, `src/packed.cu`, `src/session_store.cu`, `src/weight_layout.cu`, `src/weight_loader.cu`), so the breakage is consistent with an incomplete refactor rather than a bad audit read.

Implication:
- The audit understated the severity. The current tree is not just missing the five CUDA model files it named; it also points base-runtime manifests at non-existent model/config/execution sources.

### 2. Ragged packed prefill is still wave-based single-token stepping
**Verdict:** `Verified on current tree`

The current concurrent CUDA engine still batches prefill in token waves, not as a flattened multi-token ragged batch.

- [src/backend/cuda/runtime/engine.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/runtime/engine.cpp:389) enters ragged packed prefill when enough requests are active.
- It computes `maximum_tokens`, iterates `wave` from `0` to `maximum_tokens - 1`, builds per-wave `models`, `page_tables`, `tokens`, and `finalize_rows`, then calls `packed_executor_->prefill_step(...)` once per wave at [src/backend/cuda/runtime/engine.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/runtime/engine.cpp:417).
- Each row contributes exactly one token to each wave via `tokens.push_back(request.prompt[item.begin + wave])` at [src/backend/cuda/runtime/engine.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/runtime/engine.cpp:409).

What is verified:
- The implementation shape described by the audit is still present on `89aa918`.

What is not statically proven:
- The size of the regression versus a flattened prefill design. That is a performance hypothesis until benchmarked.

### 3. CUDA sampling still rescans the vocabulary once per requested rank
**Verdict:** `Verified on current tree`

The current sampling kernel still performs repeated full-vocabulary scans for top-k sampling.

- In [src/backend/cuda/kernels/kernels.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/kernels/kernels.cu:1279), `fused_sample_topk_kernel` loops `for (int rank = 0; rank < top_k; ++rank)`.
- Inside each rank, it scans `for (int i = threadIdx.x; i < vocab; i += blockDim.x)` at [src/backend/cuda/kernels/kernels.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/kernels/kernels.cu:1282), then masks the chosen score with `-FLT_MAX` at [src/backend/cuda/kernels/kernels.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/kernels/kernels.cu:1309).
- The packed variant repeats the same structure in `packed_sample_topk_kernel` at [src/backend/cuda/kernels/kernels.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/kernels/kernels.cu:1401) and [src/backend/cuda/kernels/kernels.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/kernels/kernels.cu:1432).

What is verified:
- The audit's algorithmic criticism is accurate: the current implementation is proportional to repeated vocab scans for non-greedy top-k sampling.

### 4. The CUDA CLI still defaults to the slower attention configuration
**Verdict:** `Verified on current tree`

The default CLI options still set `top_k = 50`, `attention_mode = "single"`, and `fast_attention = false`.

- See [src/app/cuda/main.cpp](/D:/ia/lfm25-cuda-cpp/src/app/cuda/main.cpp:38), [src/app/cuda/main.cpp](/D:/ia/lfm25-cuda-cpp/src/app/cuda/main.cpp:50), and [src/app/cuda/main.cpp](/D:/ia/lfm25-cuda-cpp/src/app/cuda/main.cpp:55).
- Those values are translated into `ModelOptions` with `AttentionMode::Single` unless the user overrides them at [src/app/cuda/main.cpp](/D:/ia/lfm25-cuda-cpp/src/app/cuda/main.cpp:383) and [src/app/cuda/main.cpp](/D:/ia/lfm25-cuda-cpp/src/app/cuda/main.cpp:404).

What is verified:
- The audit is correct that the production CLI defaults still favor the non-fast path.

What is only partially verified:
- The audit's "strict attention reads the key cache three times" claim was not revalidated against the current attention kernel bodies in this pass. The default-path part is confirmed; the exact pass count remains unverified here.

### 5. MoE residency still forces host readbacks and stream synchronizations
**Verdict:** `Verified on current tree`

The current expert-residency path still performs CPU-visible readbacks and hard stream barriers in `resolve_on_device`.

- It copies `cold_count` to host and immediately synchronizes at [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:244) and [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:246).
- Even in the all-resident fast path, it reads back the selected experts and possibly the full score array, then synchronizes again at [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:252), [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:260), and [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:264).
- The cold path also reads back the cold list and full score array, then synchronizes at [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:270), [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:282), and [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:287).
- Promotion also republishes pointer-table entries and syncs the device mirror on each promoted expert at [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:322), [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:333), and [src/backend/cuda/moe/expert_residency.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/expert_residency.cu:335).

What is verified:
- The audit's structural criticism is still current.

### 6. The current MoE router still uses serial hidden-dimension dot products and a quadratic top-k pattern
**Verdict:** `Verified on current tree`

The current router kernel is still a custom kernel with per-expert serial accumulation across `hidden_dim`, not a GEMM-backed router.

- `moe_router_kernel` loops over experts and computes `for (int h = 0; h < hidden_dim; ++h)` inside each expert at [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:43) and [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:46).
- Its own comment says the selection is `O(E*K)` at [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:22), but the actual code nests `k`, `e`, and `t` loops at [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:62), [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:66), and [src/backend/cuda/moe/route.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/route.cu:68), which is worst-case `O(E*K^2)`.

What is verified:
- The audit is correct that the comment understates the selection complexity.
- The router is still not using a cuBLAS/cuBLASLt GEMM path.

### 7. Prefix KV copy-on-write still performs many synchronous device-to-device copies
**Verdict:** `Verified on current tree`

The CUDA paged-KV prefix clone path still loops over attention layers and issues synchronous `cudaMemcpy` calls per layer.

- `clone_page_prefix(...)` is implemented in [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:86).
- In INT8 mode it performs four synchronous copies per layer at [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:111), [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:114), [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:119), and [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:122).
- In BF16 mode it performs two synchronous copies per layer at [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:130) and [src/backend/cuda/memory/paged_kv.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/memory/paged_kv.cu:133).

What is verified:
- This part of the audit is still current.

### 8. The packed metadata churn and decode-step synchronization claims are real for `6011b79`, but not currently attributable to `89aa918`
**Verdict:** `Verified, but historical/stale`

The pasted audit describes the old packed executor accurately, but the current tree does not contain the refactored `src/backend/cuda/model/packed.cu` implementation it would need for a current-master attribution.

Historical evidence from `6011b79`:
- `copy_metadata(...)` rebuilds row-wise positions, temperatures, penalties, top-k, top-p, logits pointers, seen pointers, RNG pointers, sampled destinations, position destinations, page tables, and per-layer pointer arrays in `src/packed.cu`.
- It then issues many H2D copies, including full `num_hidden_layers * maximum_batch` pointer tables, as shown in `6011b79:src/packed.cu`.
- `decode(...)` performs D2H copy of sampled tokens and a full `cudaStreamSynchronize` before returning in `6011b79:src/packed.cu`.
- `prefill_step(...)` also ends with `cudaStreamSynchronize` in `6011b79:src/packed.cu`.

What is verified:
- The audit's description matches the historical code.

What is not attributable to `89aa918`:
- The current tree's intended replacement files are absent, so this cannot honestly be presented as a property of the current implementation.

### 9. The MoE FFN criticism is partly stale and partly still directionally valid
**Verdict:** `Incorrect or overstated`

The current MoE FFN code is not the exact scalar-like version described in the audit.

- The current implementation in [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:17) explicitly documents improvements over an older version.
- It stages hidden activations into shared memory and tiles channels in `moe_gate_up_swiglu_tiled_kernel` at [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:22), [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:73), and [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:91).
- It accumulates expert contributions in FP32 in `moe_down_tiled_kernel` and `launch_finalize_moe_output` at [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:96) and [src/backend/cuda/moe/ffn.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/moe/ffn.cu:141).

What remains directionally valid:
- The current FFN still uses custom kernels, not grouped GEMM or tensor-core MoE dispatch.

What is overstated:
- The audit treats the present code as if none of the tiling/shared-memory/FP32-accumulation improvements exist.

### 10. The CPU backend notes are mixed: some are current, some are stale
**Verdict:** `Verified on current tree`

Current CPU issues confirmed by code:
- `cpu_rmsnorm_inplace` still allocates a temporary vector on every call at [src/backend/cpu/kernels/kernels.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/kernels/kernels.cpp:460).
- `cpu_qk_norm_rope` still rebuilds `cos_vals` and `sin_vals` per call at [src/backend/cpu/kernels/kernels.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/kernels/kernels.cpp:493) and [src/backend/cpu/kernels/kernels.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/kernels/kernels.cpp:501).
- CPU GQA still allocates `scores` per invocation in both FP32 and BF16 paths at [src/backend/cpu/kernels/kernels.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/kernels/kernels.cpp:539) and [src/backend/cpu/kernels/kernels.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/kernels/kernels.cpp:580).

Where the audit is stale:
- The claim that packed prefill is effectively token-by-token does not apply to the current CPU backend. The CPU path has a chunked prefill implementation using GEMM in [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:632), [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:672), [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:698), [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:726), and [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:733).
- `CpuInferenceSession::prefill(...)` switches to chunked prefill above the threshold at [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:923), [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:929), and [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:935).
- CPU sampling also uses `std::partial_sort` over a single full-vocab index array, not repeated full rescans per rank, at [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:774) and [src/backend/cpu/model.cpp](/D:/ia/lfm25-cuda-cpp/src/backend/cpu/model.cpp:778).

### 11. The scheduler note is correct but low priority
**Verdict:** `Verified on current tree`

The batch planner still filters active lanes and uses `std::stable_sort`.

- See [src/runtime/concurrency/batch_planner.cpp](/D:/ia/lfm25-cuda-cpp/src/runtime/concurrency/batch_planner.cpp:13), [src/runtime/concurrency/batch_planner.cpp](/D:/ia/lfm25-cuda-cpp/src/runtime/concurrency/batch_planner.cpp:20), and [src/runtime/concurrency/batch_planner.cpp](/D:/ia/lfm25-cuda-cpp/src/runtime/concurrency/batch_planner.cpp:32).

What is verified:
- The `O(B log B)` planner observation is correct.

What is not established here:
- The audit's downstream claim about context-length bucketing impact is plausible, but not directly demonstrated by this file alone.

### 12. The quantized-kernel recommendation is plausible, but the audit overstates what static inspection can prove
**Verdict:** `Plausible but unproven from static inspection`

The audit recommends dedicated tensor-core INT8/INT4 paths. That recommendation is plausible, but static inspection here does not support the stronger claim that current quantized kernels are definitively leaving a specific amount of hardware utilization on the table.

What is visible:
- The backend already has a cuBLASLt dispatcher for GEMM plan selection in [src/backend/cuda/runtime/gemm_dispatcher.cu](/D:/ia/lfm25-cuda-cpp/src/backend/cuda/runtime/gemm_dispatcher.cu:52).
- This pass did not establish an equivalent tensor-core-backed specialized path for the custom quantized decode kernels.

What remains unproven:
- The magnitude of the quantized-kernel opportunity without profiler evidence.

## Unsupported Claims That Need Measurement

These claims are directionally reasonable, but a static code audit cannot elevate them to facts:

- "Largest likely gains" ordering across ragged prefill, sampling, attention, MoE, and residency.
- Any estimate that a replacement design "probably" dominates the payoff without timing data.
- Any utilization claim that depends on warp occupancy, memory throughput, tensor-core issue rate, launch overhead, or graph-capture behavior.
- Any claim that one implementation "rivals lightweight kernels" or "destroys pipelining" in practice without Nsight or benchmark evidence.

## Residual Uncertainty Caused by the Broken Current Tree

- The missing `src/backend/cuda/model/*.cu` files prevent a clean current-tree review of packed decode internals after the refactor.
- The missing `src/model/...` subtree means the branch cannot be treated as a coherent runtime baseline for model/config/execution code.
- Several claims in the pasted audit clearly describe `6011b79` accurately, but they cannot be cleanly projected onto `89aa918` until the refactor is repaired and the intended replacement files are present.
