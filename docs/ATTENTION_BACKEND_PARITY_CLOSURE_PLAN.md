# Attention Backend Parity Closure Plan

## Goal

Close the remaining CPU and Metal attention/position/state capability gaps relative to CUDA without forcing the three backends to share execution mechanisms that should remain architecture-specific.

The target is semantic and execution-mode parity where CUDA already has a real implementation. Performance parity is a second layer: once a feature is semantically correct in every backend, remove avoidable token-wise fallbacks and add optimized backend-native paths where measurement justifies them.

This plan complements `docs/METAL_MROPE_GEOMETRY_PLAN.md`. That document remains the detailed implementation plan for Metal sectioned MRoPE; this document owns the broader closure order and cross-backend gates.

## Scope

In scope:

- attention pattern parity;
- RoPE/MRoPE geometry and scaling parity;
- attention-state scalar/layout parity;
- value normalization parity;
- ordinary, projected-latent and factorized-latent attention execution;
- token/decode, batched decode, prefill and packed/chunk execution coverage where the backend exposes those modes;
- backend capability declarations and anti-regression tests;
- cross-backend differential fixtures;
- removal of avoidable semantic fallbacks after correctness is established.

Out of scope for this closure plan:

- external-memory support on CUDA/Metal, because CUDA does not currently provide it and therefore it is not a CPU/Metal-vs-CUDA parity gap;
- latent MRoPE, because CUDA also rejects it today;
- new attention semantics not already represented by the IR;
- forcing CPU, CUDA and Metal to use identical reduction, paging, vectorization or synchronization mechanisms;
- MoE payload layouts that CUDA itself does not support;
- performance claims without benchmark evidence.

## Baseline truth

The executable capability declarations are the source of truth, not the documentation matrix.

The table below records the current executable state after closure Stages 0-4:

| Capability | CPU | CUDA | Metal | Closure action |
| --- | --- | --- | --- | --- |
| Full causal | yes | yes | yes | regression only |
| Sliding window | yes | yes | yes | regression only |
| Bidirectional | yes | yes, constrained | yes, constrained | declared dense no-bias scope closed; regression only |
| Prefix-LM | yes | yes, constrained | yes, constrained | declared dense no-bias scope closed; regression only |
| BlockSparse | yes | yes, constrained | no | implement Metal |
| DynamicSparse | yes, constrained | yes | no | implement Metal |
| ALiBi | yes | yes | yes | regression only |
| Relative-position bias | yes | yes, constrained | yes | differential coverage; do not overclaim CUDA combinations |
| Standard RoPE | full IR surface on CPU | broad scaling surface | full-width unscaled only | expand Metal |
| Three-axis MRoPE | interleaved + sectioned geometry | interleaved + sectioned geometry | interleaved + sectioned geometry | geometry parity closed; scaling/partial-width remain Stage 5 |
| Value RMSNorm before KV store | yes | yes | yes | parity closed; regression only |
| Ordinary BF16 KV | yes | yes | yes | regression only |
| Ordinary INT8 KV | no | yes | no | implement CPU and Metal |
| General ordinary KV layout/paging surface | yes | yes | partial/internal | formalize Metal before INT8/general sparse state work |
| Projected latent attention | yes | yes | no | implement Metal |
| Factorized latent attention | yes | yes | no | implement Metal |
| Shared KV publisher/consumer | yes | yes | yes | regression only |
| Current-value orthogonalization | yes | yes | yes | regression only |
| Output gates on representable ordinary surface | yes | yes | yes | regression only |

Historical baseline correction: `docs/ATTENTION_IR_COVERAGE.md` previously marked CPU DynamicSparse as implemented even though `CpuModelCompiler` advertised `.dynamic_sparse = false`. Stage 0 reconciled the documentation with executable capability truth. Stage 3 subsequently added the scoped standard-attention ordinary-KV implementation and flipped the executable CPU capability only after production execution and tests existed.

## Design rules

### 1. Capability flags follow implementation

Never flip a backend capability to `true` before all required runtime bindings and at least one execution-level test exist.

The order for every new cell is:

```text
canonical semantics
    -> backend semantic mirror/lowering if needed
    -> runtime/state ownership
    -> production execution
    -> differential test
    -> capability advertisement
    -> documentation matrix
```

### 2. Share mathematics, not hardware scheduling

Canonical host/CUDA semantic helpers remain appropriate for pure rules such as visibility, sparse selection, RoPE geometry and attention-state transitions.

Metal may mirror those tiny functions in MSL with conformance probes. CPU keeps vectorized/NUMA execution; CUDA keeps warp/shared-memory specialization; Metal keeps simdgroup/threadgroup specialization.

### 3. Preserve numerical contracts

Do not casually centralize:

- BF16/INT8 dequantization association;
- FMA-sensitive accumulation ordering;
- online vs three-pass attention algorithms;
- backend-specific reduction trees;
- cache addressing for physically different layouts.

Any helper extraction touching those areas requires before/after numerical fixtures.

### 4. Semantic parity before fast-path parity

A correct token-wise fallback is acceptable temporarily. Once the feature is proven, add chunk/packed/cooperative paths separately and benchmark them.

### 5. No undocumented partial support

If a feature is supported only for BF16, standard attention, prefill, a maximum selected-block count, or another bounded surface, encode the restriction in capability validation and tests.

## Stage 0 — Make backend capability truth executable

**Status: complete.**

### Work

1. Correct the stale CPU DynamicSparse cell in `docs/ATTENTION_IR_COVERAGE.md`.
2. Add `value_norm` as an explicit row in the coverage matrix.
3. Replace ad-hoc aggregate assumptions with a table-driven backend capability fixture that asks the same feature questions of CPU, CUDA and Metal.
4. Keep semantic capability separate from execution-mode capability. A backend may support a feature in standard attention but reject it for latent or packed execution.
5. Add a small generated/reporting helper or test data structure that can be used to update the documentation matrix from the same declared truth without making Markdown the source of truth.

### Acceptance criteria

- CPU DynamicSparse is reported unsupported until its production implementation lands;
- Metal value norm is reported unsupported until implemented;
- every existing `AttentionBackendCapabilities` field has an explicit backend expectation in tests;
- a newly added capability field causes the parity test to require decisions for CPU, CUDA and Metal.

## Stage 1 — Complete Metal MRoPE geometry parity

**Status: complete for the geometry scope.** Scaling, partial-width rotation, alternate theta/pairing/axis-count semantics remain intentionally assigned to Stage 5.

Execute `docs/METAL_MROPE_GEOMETRY_PLAN.md` fully.

### Required result

- Metal MSL geometry mirror exists and is probe-tested against `src/celeg/model/rope_geometry.hpp`;
- `interleaved` is carried explicitly through all Metal MRoPE runtime ABIs;
- token/decode, fused Q/K-norm + KV-store and batch MRoPE paths use the same layout semantics;
- sectioned three-axis MRoPE is accepted only after kernel-level CPU-vs-Metal parity is demonstrated;
- current interleaved output does not regress.

Do not combine this with RoPE scaling or partial rotary width. Geometry must be closed first.

## Stage 2 — Metal value normalization quick win

**Status: complete.** CPU, CUDA and Metal now advertise `value_norm = true` for their declared ordinary-attention surfaces.

### Implemented result

1. Metal applies V RMSNorm immediately after the locally owned value projection and before every KV publication path.
2. Per-head V normalization reuses `celeg_head_rmsnorm_inplace` / `celeg_head_rmsnorm_batch_inplace`; whole-vector normalization reuses the existing RMSNorm token/batch kernels. No Q/K-specific shader ABI was widened.
3. `CompiledAttentionExecution::has_key_value` is the ownership boundary: private attention and shared-KV publishers normalize V; shared-KV consumers neither project nor normalize publisher-owned V.
4. `AttentionValueNorm` loading uses the semantic width: `head_dim` for `PerHead`, `key_value_width()` for `WholeVector`. Weighted `Scale`, `OnePlusScale`, and weightless `None` semantics therefore use the existing weight-plan/materialization contract.
5. Resolved value-norm tensors are width-checked before device execution, and the Metal capability validator rejects invalid epsilon before dispatch.
6. Token/decode and batched-prefill paths normalize before both fused and standalone KV publication paths.
7. `metal_value_norm_test` uses `cpu_qk_norm_only` as the executable CPU oracle for token/batch × PerHead/WholeVector, verifies the published cache, and uses a deliberately distinguishable twice-normalized reference to guard against accidental double normalization.
8. `attention_norm_weight_plan_test` covers weighted, `OnePlusScale`, weightless, and shared-consumer ownership; capability tests cover the supported norm surface and malformed epsilon.
9. `.value_norm = true` and the cross-backend capability matrix were flipped only after production execution and the differential harness existed.

### Acceptance criteria

- Metal output and stored V match CPU/CUDA semantic expectations within the established float tolerance;
- value cache contains normalized V exactly once;
- consumer layers never re-normalize publisher-owned V;
- capability tests reject malformed norm widths/weights before dispatch.

## Stage 3 — Implement CPU DynamicSparse

**Status: complete for the declared standard-attention ordinary-KV surface.** SIMD-specialized block scoring remains a Stage 13 performance follow-up; latent and INT8 state combinations are not implied by this status.

The canonical content-ranked selection contract in `src/celeg/attention/dynamic_sparse_semantics.hpp` remains the semantic owner shared with CUDA.

### Implemented result

1. CPU now has a production DynamicSparse executor for standard attention over ordinary paged KV state.
2. Selection preserves the canonical CUDA/host contract: only causal candidate blocks are considered; each block score is the maximum scaled Q·K over its visible tokens; top-K insertion is deterministic; ties prefer the lower block index.
3. The implementation supports CPU FP32 and BF16 ordinary KV storage. CPU INT8 remains an independent Stage 8 gap.
4. The selected blocks feed the normal attention computation using the CPU online-softmax semantics; DynamicSparse is not represented as or silently degraded to a position-only mask.
5. Decode and paged prefill use the same semantic selection contract. The parallel paged entry point falls back explicitly to the correct semantic path rather than entering an incompatible dense/parallel fast path.
6. `cpu_dynamic_sparse_attention_test` checks selected blocks and final output against a dense host oracle for FP32/BF16, negative scores, ties, partial/tail blocks, nontrivial later winners, decode, and paged prefill.
7. Shared canonical DynamicSparse semantic tests cover the selection primitives used by CPU and CUDA; the dedicated CUDA semantic fixture remains separate from structural pattern visibility tests.
8. Compiler/capability validation constrains CPU DynamicSparse to standard attention plus ordinary KV and rejects unsupported execution/state combinations before dispatch.
9. CPU `.dynamic_sparse = true` and the coverage matrix were updated only after the production path and execution-level tests were present.

### Acceptance criteria

- CPU selected blocks agree with the canonical selection contract;
- CPU output agrees with a dense masked oracle across FP32/BF16 fixtures;
- negative scores, deterministic ties, partial blocks, and nontrivial winning blocks are covered;
- decode and paged prefill execute the same content-ranked semantics;
- parallel/chunked entry points either execute the feature correctly or fall back explicitly without semantic drift;
- no latent, INT8, or SIMD fast-path support is inferred from the aggregate capability flag.

## Stage 4 — Add Metal dense non-causal pattern support

**Status: complete for the declared standard-attention ordinary-BF16 no-bias surface.** Biased Bidirectional/Prefix-LM combinations remain explicit rejections rather than being approximated by causal or biased kernels.

### Implemented result

1. `pattern_semantics.hpp` now owns explicit bidirectional visibility and Prefix-LM visible-sequence-length semantics; CPU consumes the same bidirectional helper so host/backend truth remains aligned.
2. `dense_pattern.metal` mirrors those narrow rules and exposes `celeg_attention_dense_pattern_semantics_probe`, which `metal_dense_pattern_semantics_test` checks exactly against the host contract at prefix boundaries and partial-availability cases.
3. Production batched prefill uses one `celeg_attention_batch_dense_pattern` kernel for both patterns. It changes only the visible span and reuses the existing `celeg_attention_span_one_exp` reduction/softmax core rather than copying the attention implementation.
4. Bidirectional prefill sets each query span to all K/V rows available in the full batch. Prefix-LM prefix queries see the complete available prefix, including future prefix keys; post-prefix queries remain causal.
5. Metal publishes all locally owned K/V rows for the prefill batch before the dense non-causal attention dispatch. Shared-KV consumers continue to address the earlier publisher-owned cache and therefore observe the publisher's complete batch state.
6. `prefill_session()` rejects Bidirectional/Prefix-LM if the runtime would fall back to token-wise prefill. Prefix-LM also rejects a prefill shorter than `prefix_length`, preventing an incomplete prefix from being silently treated as complete.
7. Decode at the current sequence tail reuses the existing dense causal kernel. `attention_pattern_semantics_test` explicitly proves that causal, Bidirectional, and post-prefix Prefix-LM visible sets are identical at the tail after the prefix has been completed.
8. The causal/sliding tiled fast path remains unchanged and is simply excluded when the dense non-causal pattern mode is active.
9. `metal_dense_noncausal_attention_test` uses zero Q·K scores and deliberately different V rows so a future-reading implementation is numerically distinguishable from causal execution: the first-row expected values differ for causal, Prefix-LM, and Bidirectional attention.
10. Metal capability validation advertises Bidirectional and Prefix-LM only after the production path and tests exist, requires positive Prefix-LM length, and keeps attention bias rejected for these two patterns. Ordinary BF16 state and standard-execution restrictions remain governed by the existing Metal capability contract.

### Acceptance criteria

- MSL dense-pattern semantics match the canonical host contract exactly;
- production batched prefill demonstrably reads future keys for Bidirectional and prefix queries;
- Prefix-LM boundary and incomplete-prefix behavior is explicit;
- decode-tail reuse is justified by an executable semantic equivalence test rather than assumption;
- token-wise prefill cannot silently produce causal semantics for a future-reading pattern;
- existing causal/sliding performance paths remain intact;
- biased dense non-causal combinations remain stable pre-dispatch rejections.

## Stage 5 — Close standard Metal RoPE parity

After MRoPE geometry is stable, expand ordinary Metal RoPE toward the CPU/CUDA surface.

### Substage 5A — Partial rotary width

- pass explicit rotary pair count/rotary dimension to Metal kernels;
- rotate only the configured prefix;
- preserve untouched tail components;
- verify Q/K normalization still covers the intended full head independently of rotary width.

### Substage 5B — RoPE scaling semantic mirror

Introduce a focused Metal scaling helper mirroring the model/CUDA semantic variants:

- No scaling;
- Linear;
- Dynamic NTK;
- YaRN;
- LongRoPE;
- Llama-3 frequency scaling;
- Proportional scaling.

Do not copy the CUDA lowering struct blindly. Define a Metal-friendly ABI with explicit scalar fields and bounded factor buffers where required.

Add an integer/float probe layer comparing Metal frequency results to `rope_frequency()` over representative pairs and positions before production kernels consume it.

### Substage 5C — Production migration

Route generic Metal RoPE/MRoPE frequency computation through the tested scaling helper while leaving trig, normalization, threadgroup shape and fused KV publication backend-specific.

### Acceptance criteria

- every scaling variant is either implemented and differential-tested or remains an explicit stable rejection;
- partial-width standard RoPE matches CPU for both SplitHalf and AdjacentPairs;
- existing full-width unscaled kernels show no unexplained regression.

## Stage 6 — Add Metal BlockSparse

BlockSparse is the lower-risk sparse feature because visibility is structural rather than content-ranked.

### Work

1. Add an MSL mirror/probe for block-sparse visibility if the existing Metal pattern mirror does not already expose it to production.
2. Preserve the model contract from `pattern_semantics.hpp`.
3. Implement a Metal sparse attention execution path that keeps the current three-pass versus online-softmax distinction explicit where numerical behavior requires it.
4. Start with ordinary BF16 KV and standard attention only, matching CUDA's constrained-pattern philosophy.
5. Cover decode and batched prefill separately.

### Acceptance criteria

- selected visible tokens match CPU/CUDA for representative local/global block configurations;
- output matches a dense masked oracle;
- unsupported bias/state combinations reject before dispatch.

## Stage 7 — Add Metal DynamicSparse

Only after BlockSparse and CPU DynamicSparse are stable.

### Work

1. Mirror the canonical content-ranked block-selection mechanics in MSL, including deterministic tie behavior.
2. Keep Metal simdgroup reduction and candidate storage backend-specific.
3. Start with the same explicit bound as CUDA (`max_selected_blocks <= 32`) unless Metal measurements or storage design justify a different tested limit.
4. Implement ordinary BF16 standard-attention prefill/decode first.
5. Add CPU-vs-CUDA-vs-Metal selected-block and output differential fixtures.

### Acceptance criteria

- all three backends select the same blocks for the same Q/K content;
- ties prefer the lower block index everywhere;
- sparse output matches a dense oracle within backend numerical tolerance;
- no fallback silently becomes BlockSparse or causal attention.

## Stage 8 — CPU INT8 ordinary KV state

CUDA already has INT8 ordinary KV support; CPU explicitly rejects it.

### Work

1. Audit the exact state-storage metadata and scale ownership used by CUDA before designing CPU storage.
2. Reuse the semantic quantization contract, not CUDA memory layout.
3. Add CPU INT8 key/value load helpers whose multiplication association is explicitly tested.
4. Add quantize/store and dequantize/read coverage for token, prefill and paged traversal.
5. Integrate with ordinary dense attention first, then sparse patterns.
6. Test shared-KV ownership with INT8 state before advertising that combination.
7. Add AVX2/AVX-512/VNNI acceleration only after scalar parity.

### Acceptance criteria

- round-trip storage tests establish quantization/scale behavior;
- BF16/FP32 paths remain unchanged;
- CPU INT8 attention agrees with CUDA/reference within a quantization-appropriate tolerance;
- no implicit widening changes the advertised state type contract.

## Stage 9 — Formalize Metal ordinary KV layout/paging before INT8

Metal currently uses an internal page-sized physical layout but does not expose the same general layout/paging capability surface as CPU/CUDA.

### Work

1. Separate semantic KV ownership from physical Metal addressing.
2. Define one Metal storage accessor contract used by dense, sparse and future latent paths.
3. Make contiguous vs paged/page-table addressing explicit where the IR/runtime distinguishes them.
4. Preserve the current optimized internal layout as one implementation, not as an implicit semantic assumption.
5. Add address/alias tests for private and shared KV, including page boundaries and prefix snapshots.

### Acceptance criteria

- Metal can state exactly which layout/paging combinations it supports;
- shared-KV consumers resolve publisher-owned storage through the same accessor contract;
- attention kernels no longer bake hidden layout assumptions that block INT8/sparse expansion.

## Stage 10 — Metal INT8 ordinary KV state

Build on Stage 9 rather than adding one-off INT8 branches to every kernel.

### Work

1. Define Metal INT8 state payload and scale buffers consistent with the semantic state contract.
2. Add store/quantize and load/dequantize helpers with conformance tests against CUDA/host behavior.
3. Extend dense causal/sliding first.
4. Extend bidirectional/Prefix-LM.
5. Extend BlockSparse/DynamicSparse only after dense INT8 is stable.
6. Extend shared-KV publication/consumption last.

### Acceptance criteria

- INT8 state is a real runtime storage mode, not BF16 storage with temporary quantization;
- no extra dequantized cache duplicates the full KV state;
- numerical tolerance and performance/memory results are measured separately.

## Stage 11 — Metal projected latent attention

Do not start factorized latent until direct projected latent state ownership is correct.

### Work

1. Mirror the existing `CompiledAttentionExecution::Latent` ownership contract.
2. Add Metal latent-state allocation, projection and cache addressing.
3. Support the same current CUDA restrictions initially: standard supported patterns, no MRoPE, no unsupported relative-bias combinations, explicit latent-rank bound.
4. Implement token/decode first, then batched prefill.
5. Reuse existing output projection and supported gate/transform ordering only where the IR permits it.

### Acceptance criteria

- Metal projected-latent output matches CPU/CUDA fixtures;
- state size and addressing are tested independently from attention output;
- unsupported combinations fail at capability validation rather than pipeline lookup.

## Stage 12 — Metal factorized latent attention

### Work

1. Implement factorized query/key/value latent projection ownership using the same IR contract as CPU/CUDA.
2. Add supported factorized output gates after ungated parity is proven.
3. Cover token/decode and batched prefill.
4. Keep latent MRoPE rejected because it remains outside CUDA parity.

### Acceptance criteria

- ungated factorized execution matches CPU/CUDA;
- supported gate granularities match the shared representation contract;
- packed combinations remain explicit if unsupported.

## Stage 13 — Remove remaining CPU semantic-performance fallbacks

After CPU capability parity is reached, close execution-mode gaps that are semantically correct but slower than CUDA-style specialized paths.

Highest-value candidates:

- factorized latent chunk prefill;
- factorized latent packed/ragged prefill;
- factorized latent packed decode;
- DynamicSparse vectorized block scoring;
- INT8 KV vectorized dot/dequant paths.

Do not replace a correct fallback until the specialized path has a differential fixture.

## Stage 14 — Cross-backend differential harness

Create a reusable tiny-attention fixture capable of running the same resolved attention case through every available backend.

The fixture should make Q/K/V and weights deterministic and small enough that a host oracle can compute expected output.

Required dimensions to vary:

- pattern;
- bias;
- position encoding;
- Q/K/V normalization;
- state scalar type;
- private/shared KV;
- standard/latent/factorized execution;
- token/decode/prefill mode;
- page boundary placement;
- GQA/MQA head mapping.

The harness must report separately:

1. semantic selection/visibility mismatch;
2. state/cache mismatch;
3. numerical output mismatch;
4. unsupported-by-contract result.

Do not use one universal floating-point tolerance. Maintain per-state/per-backend tolerances and keep exact integer semantic probes exact.

## Stage 15 — Capability matrix anti-regression gate

Once the new features land:

1. update `ATTENTION_IR_COVERAGE.md` from tested truth;
2. require each backend to declare every capability field;
3. keep constrained combinations in table-driven tests;
4. add CI jobs that at least compile CPU and CUDA capability fixtures everywhere and run Metal fixtures on eligible macOS runners;
5. require the normal generated Metal inference shader to compile, not only isolated probe fragments;
6. keep performance benchmarks outside semantic pass/fail gates unless a specific regression threshold is intentionally adopted.

## Recommended implementation order

The order below minimizes architectural rework and gets useful parity quickly:

```text
0. capability truth / stale docs                    [complete]
1. Metal sectioned MRoPE                            [complete]
2. Metal value norm                                 [complete]
3. CPU DynamicSparse                                [complete]
4. Metal Bidirectional + Prefix-LM                  [complete]
5. Metal partial/scaled RoPE
6. Metal BlockSparse
7. Metal DynamicSparse
8. CPU INT8 KV
9. Metal storage/layout abstraction
10. Metal INT8 KV
11. Metal projected latent
12. Metal factorized latent
13. CPU fast-path closure
14. cross-backend differential harness hardening
15. final capability/CI gate
```

Some remaining work can proceed in parallel after Stage 4:

- CPU INT8 is independent of Metal shader work;
- Metal RoPE scaling is mostly independent of Metal sparse execution;
- the differential harness can grow incrementally with every stage rather than waiting until Stage 14.

## Suggested commit discipline

Each capability should land in reviewable slices:

```text
semantics/probe
runtime ownership or ABI
production kernel/path
execution-level tests
capability flag
coverage docs
optional optimization
```

Do not combine the capability flag and the first implementation attempt in one large commit. This keeps intermediate states honest and makes bisecting semantic regressions practical.

## Completion criteria

This closure plan is complete when:

- every attention feature CUDA currently supports has either an equivalent CPU/Metal implementation or a documented architecture-specific reason it cannot sensibly apply;
- CPU supports DynamicSparse and ordinary INT8 KV;
- Metal supports sectioned MRoPE, value norm, bidirectional, Prefix-LM, BlockSparse, DynamicSparse, partial/scaled standard RoPE, ordinary INT8 KV, projected latent and factorized latent attention for explicitly declared modes;
- Metal's layout/paging contract is explicit rather than hidden in kernel addressing;
- backend capability declarations, production execution and documentation agree;
- unsupported combinations fail before execution with stable diagnostics;
- cross-backend differential tests cover each newly closed cell;
- specialized fast paths never replace a correct fallback without parity tests;
- no claim of performance improvement is made without measured benchmark evidence.