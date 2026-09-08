# CUDA Attention Algorithm Inventory

This inventory classifies the current CUDA attention implementations by numerical algorithm, storage layout, and orthogonal policy before any further DRY extraction. The goal is to share only primitives whose numerical and synchronization contracts are genuinely identical while preserving tuned launch geometry and specialized kernels.

## 1. Classification rules

Attention implementations are considered the same algorithm only when they preserve the same:

- softmax formulation and accumulation order;
- BF16 rounding points;
- synchronization contract;
- visibility semantics;
- KV addressing contract;
- output accumulation precision;
- launch assumptions relevant to generated code.

Source similarity alone is not sufficient reason to merge kernels.

## 2. Dense strict attention

Primary implementations:

- `attention_dense.cuh` for contiguous BF16/INT8 paths;
- `attention_paged.cuh` for paged BF16/INT8 paths;
- batch-pointer strict variants where selected by the capability layer.

Numerical contract:

1. first pass computes the row maximum;
2. second pass computes the denominator;
3. third pass recomputes scores and accumulates probability-weighted values.

Strict attention intentionally rounds score/probability intermediates through BF16-compatible semantics. This is algorithm-defining and must not be merged with online attention.

Contiguous, paged, and pointer-based launchers may share small primitives, but their KV addressing and launch geometry remain separate responsibilities.

## 3. Dense online attention

Primary implementations:

- contiguous BF16/INT8 online paths in `attention_dense.cuh`;
- paged BF16/INT8 online paths in `attention_paged.cuh`;
- batch-pointer online paths.

The stable online state transition is:

```text
next_max = max(running_max, score)
alpha = exp(running_max - next_max)
beta = exp(score - next_max)
denominator = denominator * alpha + beta
accumulator = accumulator * alpha + value * beta
running_max = next_max
```

This numerical contract is distinct from strict three-pass attention. A future shared online-step primitive is acceptable only if compiler output, register pressure, and runtime performance remain equivalent.

## 4. Segmented attention

Primary implementations:

- contiguous decode partial/reduce in `attention_segmented.cuh`;
- contiguous prefill partial/reduce in `attention_segmented.cuh`;
- paged decode BF16/INT8 partial/reduce in `attention_paged.cuh`.

Segment partial kernels use the online softmax state transition. Segment reduction then combines independent partials through a log-sum-exp merge:

```text
global_max = max(partial_max)
factor = exp(partial_max - global_max)
denominator += partial_denom * factor
accumulator += partial_accum * factor
output = accumulator / denominator
```

The reduce loop order, precision, and zero-denominator skipping are identical across contiguous decode, contiguous prefill, and paged decode. This merge is the first approved Phase 3 shared primitive candidate.

The partial kernels and launch geometry remain specialized.

Existing CUDA runtime tests already cover:

- contiguous segmented decode against online decode across multiple segment counts;
- paged BF16 segmented decode against the non-segmented paged path;
- paged INT8 segmented decode against the non-segmented paged path.

## 5. Block-sparse attention

Implementations include contiguous prefill, contiguous decode, and paged decode.

These paths use strict three-pass softmax with BF16-rounded score/probability semantics plus a visibility predicate. Their layout and visibility rules differ from dense strict attention even though the numerical softmax contract is related.

Keep the sparse kernels specialized. Share only small visibility/addressing or strict-softmax primitives when they remain semantically exact and do not obscure the sparse policy.

## 6. Dynamic sparse attention

Dynamic sparse attention performs a selection stage before strict BF16-rounded three-pass attention.

Selection is algorithm-defining state and must remain outside any generic dense/block-sparse kernel abstraction. Do not collapse dynamic and static block-sparse attention into a mega-template.

## 7. ALiBi and relative-bias attention

ALiBi and relative-bias paths use the same online-softmax family as dense online attention. Their independent variation is score construction:

- ALiBi adds a head-dependent distance term;
- relative bias adds a bucketed position-dependent bias.

These are valid OCP policy candidates because score bias is orthogonal to online accumulation. However, contiguous/paged/pointer launchers and KV addressing remain specialized.

Any future score-policy extraction must be trivially inlinable and must not increase register pressure or introduce runtime policy branches.

## 8. Flash prefill

`attention_gemm.cuh` contains the tiled Flash-style prefill kernel.

It uses shared-memory Q/K/V tiles and tile-level online rescaling. Its memory staging, tile geometry, register accumulation, and warp XOR reductions are algorithm/performance defining.

Do not route Flash through the ordinary online attention abstraction. In particular, do not replace its XOR reductions mechanically with canonical reductions because floating-point reduction order and generated code differ.

## 9. GEMM prefill

The GEMM path uses:

1. cuBLAS QK multiplication;
2. a separate causal softmax kernel;
3. cuBLAS probability/value multiplication.

This materialized-score algorithm is intentionally separate from direct strict/online kernels and must remain specialized.

## 10. Latent attention

`attention_latent.cuh` owns factorized latent query/value transforms, optional rotary state, latent KV addressing, and an online-softmax update over latent-rank outputs.

Although its online update resembles ordinary GQA online attention, latent geometry and state semantics are independently meaningful. Keep latent attention as a separate family. Consider sharing only a smaller scalar online transition after profiling proves that doing so is code-generation neutral.

## 11. Capability layer

`attention_capability.hpp` already exposes the important algorithm choices explicitly:

- `Strict`;
- `Online`;
- `Segmented`;
- `Flash`;
- `Gemm`;
- `Alibi`;
- `RelativeBias`.

Important current constraints:

- BF16 prefill fast policy prefers Flash/GEMM rather than ordinary Online;
- INT8 prefill fast uses Online;
- long BF16 prefill may select Segmented after the GEMM threshold;
- decode BF16/INT8 contiguous and paged support Strict/Online/Segmented;
- batch-pointer decode supports Strict/Online but not Segmented;
- ALiBi/relative bias select their dedicated bias algorithm family.

The capability matrix remains the source of truth for which specialized launcher is legal. Kernel extraction must not move architecture/algorithm switches into operator code.

## 12. Approved extraction order

### 12.1 First: segmented partial merge

Extract only the identical segment-reduction primitive into `attention_common.cuh` and route these reduce kernels through it:

- `gqa_decode_segment_reduce_kernel`;
- `gqa_prefill_segment_reduce_kernel`;
- `gqa_decode_segment_reduce_batch_kernel`.

Requirements:

- preserve the two-loop merge order exactly;
- preserve `float` intermediates;
- preserve `local_denom == 0.0f` skipping;
- preserve launcher block sizes and grids;
- force-inline the helper;
- do not alter partial kernels.

### 12.2 Later: online state transition

Only after inspecting generated code/register pressure should ordinary online variants consider sharing a scalar state-update primitive.

### 12.3 Later: bias score policies

ALiBi and relative-bias score construction may become small compile-time/inlined policies if this reduces duplication without creating a universal attention template.

## 13. Explicit non-goals

Do not:

- build one universal attention mega-template;
- merge strict and online numerical contracts;
- hide paged/contiguous/sparse addressing behind runtime virtual dispatch;
- fold Flash/GEMM into the ordinary online kernel family;
- generalize latent attention solely because its softmax update looks similar;
- change launch geometry as part of a DRY-only extraction;
- change BF16 rounding points in strict/sparse attention.

## 14. Validation gate

Every attention code extraction must include:

1. CUDA compile verification;
2. existing capability tests;
3. segmented/runtime parity tests where applicable;
4. CUDA smoke coverage;
5. numerical comparison against the unchanged path/reference;
6. generated-code/register-pressure inspection for hot-kernel helpers;
7. focused benchmark/profile comparison before claiming performance equivalence.

The repository currently has functional segmented parity coverage but no dedicated attention-segment-reduce microbenchmark. Until a focused measurement is available, a helper extraction may be accepted as numerically/structurally correct but must not be described as proven performance-neutral solely from CI compilation.
