# Metal All-Models Closure Plan

## Goal

Turn the Apple M5 all-model smoke matrix into an executable closure program for the Metal backend. Every currently failing target model must end in one of two states:

1. correct Metal execution with a permanent regression gate; or
2. a deliberately scoped format/feature decision with an explicit diagnostic and documented rationale.

For the current matrix, the intended target is to make every listed model executable rather than treating today's unsupported-feature diagnostics as the final state.

This document is a model-driven overlay on `docs/ATTENTION_BACKEND_PARITY_CLOSURE_PLAN.md`. It does not replace the semantic backend roadmap. It connects concrete model failures to the semantic, loader, quantization, state-layout, and diagnostic work required to close them.

## Apple M5 baseline

The baseline was produced with `celeg-metal-run`, prompt `Hello`, on Apple M5.

| Model / format | Baseline | Classification | Closure lane |
| --- | --- | --- | --- |
| LFM2.5-350M safetensors | generates | regression canary | preserve |
| LFM2.5-350M GGUF BF16 | generates | regression canary | preserve |
| LFM2.5-350M GGUF F16 | generates | regression canary | preserve |
| LFM2.5-350M GGUF Q4_0 | generates | regression canary | preserve |
| LFM2.5-350M GGUF Q4_K_M | generates | regression canary | preserve |
| LFM2.5-350M GGUF Q6_K | generates | regression canary | preserve |
| LFM2.5-350M GGUF Q8_0 | generates | regression canary | preserve |
| LFM2.5-350M GGUF QAD-Q4_0 | generates | regression canary | preserve |
| LFM2.5-350M GGUF Q5_K_M | empty output, exit 0; CPU same file works | **P0 correctness defect** | Metal quantization |
| LFM2.5-8B-A1B MoE | generates | regression canary | preserve |
| Qwen3.5-0.8B | rejected: M-RoPE theta must be 10000 | missing supported semantic surface | M-RoPE generalization |
| gemma-4-E4B-it | rejected: RoPE scaling unsupported | missing supported semantic surface | standard RoPE scaling |
| Ling-3.0-tiny | rejected: factorized latent attention unsupported | missing execution/state surface | latent attention |
| gpt-neox-20b | rejected: no checkpoint format matches | loader/checkpoint gap | checkpoint ingestion |
| Qwen3.6-27B-MLX-4bit | rejected: unsupported dtype | format/quantization gap | MLX 4-bit |
| Qwen3.8-27B-4bit | rejected: unsupported dtype | format/quantization gap | MLX 4-bit |

The working LFM rows and LFM MoE are permanent Metal regression canaries. New work must not trade those successes for support of another family.

## Definition of done

The all-model closure is complete when:

- every target in the matrix loads and generates a bounded non-empty response on the M5 acceptance machine;
- no target succeeds only through an undocumented semantic approximation;
- unsupported intermediate states fail with a precise capability, format, or dtype diagnostic;
- a successful smoke result requires the requested generation work to produce at least one accepted output token rather than merely returning process exit code 0;
- every newly implemented semantic path has a compact synthetic/reference differential test independent of the large real model;
- large model runs remain acceptance tests, not the only correctness oracle;
- existing LFM dense, quantized, QAD, and MoE canaries stay green;
- hosted CI builds compact fixtures and backend targets; the physical Apple M5 suite remains the real-device Metal runtime gate where hosted runners do not execute Metal tests;
- capability flags are opened only after production execution and a differential fixture exist.

## Closure order

Recommended order:

```text
0. Harden all-model truth / smoke classification
1. Fix Q5_K_M Metal correctness
2. Close RoPE scaling + generalize M-RoPE theta
3. Add GPT-NeoX checkpoint ingestion
4. Close Metal projected/factorized latent attention for Ling
5. Add MLX 4-bit ingestion + Metal execution
6. Run and lock the final all-model M5 matrix
```

The order is deliberate:

- improve the truth signal before debugging any model;
- fix the one known correctness defect in an already-supported family before adding new capability;
- finish the already-started Stage 5 work next, unlocking Gemma and Qwen with relatively local semantic changes;
- keep checkpoint loading isolated from attention work;
- perform the larger latent-state architecture work after the simpler model-family blockers are closed;
- treat MLX 4-bit as a distinct format/quantization contract rather than pretending it is ordinary GGUF Q4;
- only then turn the matrix into a release gate.

GPT-NeoX loader work may proceed in parallel with the RoPE lane because it is mostly independent, provided it does not cause checkpoint-format changes to be mixed into attention commits.

---

## Phase 0 — Harden `celeg-metal-run` and the all-model harness

**Priority: P0 enabling work.**

The Q5_K_M result exposed a weakness in the current smoke contract: `exit 0` plus zero generated output is not a valid successful generation test.

### 0.1 Manifest-driven model matrix

Create one machine-readable manifest for the M5 suite. Each entry should record at least:

- stable model identifier/path alias;
- container/checkpoint kind: safetensors, GGUF, MLX;
- architecture/family;
- dtype or quantization label;
- prompt;
- bounded generation token count;
- expected current status during migration;
- acceptance model tag such as `regression-canary`, `feature-acceptance`, or `format-acceptance`.

Do not infer semantics from filenames such as `_M`; inspect actual tensor metadata when correctness depends on the concrete per-tensor quantization types.

### 0.2 Result classification

The harness should classify each run, rather than flattening everything into process success/failure:

```text
PASS_GENERATED
FAIL_EMPTY_GENERATION
FAIL_NONFINITE
UNSUPPORTED_CAPABILITY
UNSUPPORTED_FORMAT
UNSUPPORTED_DTYPE
LOAD_ERROR
RUNTIME_ERROR
PROCESS_ERROR
```

Preserve the backend's precise diagnostic in the report.

### 0.3 Empty-generation contract

Add a smoke-only contract around `celeg-metal-run` such as a wrapper assertion or explicit CLI option that requires generation to make progress.

Do not globally redefine an EOS-first model result as invalid for every API. The stricter rule belongs to the controlled smoke test where the fixture and generation budget are intentionally chosen to require at least one generated token.

Capture, where practical:

- generated-token count;
- first generated token id;
- whether logits contained NaN/Inf;
- first-token/top-logit fingerprint for deterministic synthetic fixtures;
- elapsed load/prefill/decode phases separately enough to distinguish a load failure from generation failure.

### 0.4 Reports

Emit both:

- JSON for CI/history/comparison;
- Markdown for the human-readable all-model table.

### Acceptance

- current Q5_K_M is red even though the process exits 0;
- every currently working LFM row is green;
- existing explicit feature/format rejections are classified correctly instead of being reported as generic failures;
- reports can be compared between commits without manually reconstructing the table.

---

## Phase 1 — Fix the Metal Q5_K_M correctness defect

**Priority: P0. This is the first production bug to fix.**

This is qualitatively different from the other red rows: the same LFM architecture and same model family work through many other Metal storage formats, while the CPU runner works on the same Q5_K_M file. The working Q4_K_M and Q6_K rows make general GGUF parsing, LFM architecture support, prompt formatting, and the overall generation loop lower-probability suspects.

Do not begin by changing sampling or chat templates.

### 1.1 Establish the actual tensor mix

Dump the GGUF tensor metadata for the failing file and record the actual quant type used by each relevant tensor.

`Q5_K_M` is a preset/file label, not proof that every matrix tensor is a single Q5_K layout. The investigation must identify the first concrete storage type and tensor whose Metal result differs from CPU/reference.

### 1.2 Block-level Q5_K oracle

Add a small CPU/reference vs Metal block differential fixture covering Q5_K decoding/dequantization semantics, including deliberately constructed cases for:

- high-bit plane (`qh`) patterns: all zero, all one, alternating, boundary bits;
- low quant-bit boundaries;
- scale/min packing and unpacking;
- signedness-sensitive values;
- vector load alignment;
- first and last sub-blocks;
- nontrivial block strides;
- values that distinguish a missing fifth bit from a wrong scale/min;
- enough blocks to cross any threadgroup/vectorized iteration boundary.

The fixture should compare the semantic dequantized values before testing a full GEMV/GEMM.

### 1.3 Matrix-operation differential

Once block dequantization is correct, compare known small matrix/vector operations through the real production Metal dispatch against the host reference.

Cover:

- dimensions exactly on the preferred tile boundary;
- dimensions one below/above the boundary;
- odd tail sizes;
- multiple output rows;
- the concrete shapes used by LFM2.5-350M where reasonably compact.

Audit specifically:

- `GGML_TYPE_Q5_K` dispatch selection;
- block byte size and stride;
- `qh` high-bit extraction;
- bit shifts and Metal integer signedness;
- scale/min nibble/bit packing;
- vectorized/aligned loads;
- tail handling;
- any specialization shared with Q4_K/Q6_K whose assumptions are invalid for Q5_K.

### 1.4 First-divergence model trace

If block and small-matrix fixtures pass, stop guessing and trace the real model CPU vs Metal until the first divergence.

Recommended checkpoints:

```text
loaded tensor metadata
    -> embedding result
    -> first layer Q/K/V projections
    -> first layer attention output
    -> first layer MLP output
    -> final norm
    -> first-token logits
```

Use hashes/statistics first and dump full tensors only around the first bad boundary.

### 1.5 Production fix and regression

Fix the lowest-level semantic owner that is actually wrong. Do not patch LFM-specific execution if the bug is generic Q5_K handling.

### Acceptance

- the exact failing LFM2.5-350M Q5_K_M file generates bounded, non-empty coherent output on Apple M5;
- the CPU/reference vs Metal Q5_K synthetic fixture passes;
- first-token logits/intermediate values match the established quantized tolerance and contain no NaN/Inf;
- Q4_K_M, Q6_K, Q8_0, Q4_0, QAD-Q4_0, F16, BF16, and safetensors LFM canaries still pass;
- LFM2.5-8B-A1B MoE still passes;
- an empty generation can no longer produce a green smoke result.

---

## Phase 2 — RoPE closure for Gemma 4 and Qwen3.5

**Priority: P1. Maps primarily to Attention Parity Stage 5B/5C.**

Stage 5A already closed ordinary unscaled partial-width RoPE. Phase 2 finishes scaling semantics and removes the unrelated hard-coded M-RoPE theta limitation.

### 2.1 Canonicalize RoPE scaling before writing more MSL

Cover every scaling alternative represented by `RopeScalingSpec`:

- no scaling;
- Linear;
- Dynamic NTK;
- YaRN;
- LongRoPE;
- Llama-3 frequency scaling;
- Proportional scaling.

Create canonical host semantic tests over representative positions and rotary pairs:

- before original context;
- exactly at the original-context boundary;
- after original context;
- first rotary pair;
- middle pair;
- last pair;
- representative full-width and partial-width configurations when the IR permits them.

### 2.2 Resolve the Dynamic NTK host/CUDA discrepancy first

The Stage 5B audit found a semantic discrepancy that must be treated as a blocker, not copied into Metal: the host helper and current CUDA path appear to apply Dynamic NTK at different points in the frequency calculation, which can make one formulation pair-dependent in a different way from the other.

Before any Metal capability is opened:

1. construct explicit numerical fixtures that distinguish the two formulas;
2. verify intended semantics against the model/checkpoint configuration contract used by Celeg;
3. choose one canonical semantic owner;
4. migrate CPU/reference and CUDA consumers to that owner/contract or document the backend-specific lowering only if it is mathematically equivalent;
5. add a regression that fails if the formulas diverge again.

Do not create a Metal mirror of a known ambiguity.

### 2.3 Metal scaling semantic mirror

Add a focused MSL helper, e.g. `rope_scaling.metal`, that mirrors the canonical semantic result rather than CUDA scheduling structures.

Use a Metal-friendly ABI:

- explicit scalar fields for bounded scalar variants;
- bounded factor buffers only where the semantic variant requires arrays;
- stable variant discriminator;
- no C++ variant-layout mirroring across the ABI.

Add a host-vs-MSL frequency probe before production kernels consume the helper.

### 2.4 Production ordinary scaled RoPE

Thread scaling through token/decode and batched-prefill production paths. Preserve `NoRopeScaling` as the existing fast path when possible.

Cover:

- SplitHalf and AdjacentPairs where semantically supported;
- full and partial rotary width;
- Q/K normalization combinations;
- published K cache contents;
- boundary positions relevant to each scaling variant.

### 2.5 Gemma 4 acceptance model

Use `gemma-4-E4B-it` as the real-model acceptance test for the ordinary scaled-RoPE lane.

The capability flag/rejection must not be relaxed merely because the checkpoint loader recognizes the model. Open only the exact scaling surface that the production kernel and differential tests cover.

### 2.6 Generalize M-RoPE theta for Qwen3.5

Treat `theta != 10000` as a separate M-RoPE parameterization gap rather than conflating it with ordinary RoPE scaling.

Actions:

- thread explicit theta through every production M-RoPE path that currently assumes 10000;
- update token/decode, fused norm+KV-store, and batched-prefill ABIs consistently;
- preserve interleaved and sectioned geometry semantics;
- add MSL-vs-host frequency/geometry cases at theta 10000 and at the concrete Qwen3.5 theta;
- preserve theta-10000 output as a regression fixture;
- only add M-RoPE scaling if a target model actually requires it; do not silently broaden the Stage 5 scope.

### 2.7 Qwen3.5 acceptance model

Use `Qwen3.5-0.8B` as the real M5 acceptance model for non-10000 M-RoPE theta.

### Acceptance

- canonical CPU/host and CUDA scaling fixtures agree for every represented variant;
- Dynamic NTK ambiguity is eliminated by executable tests;
- MSL frequency probes agree with the canonical host helper;
- `gemma-4-E4B-it` generates on M5;
- `Qwen3.5-0.8B` generates on M5;
- existing ordinary no-scaling and theta-10000 M-RoPE paths remain unchanged within their established tolerances;
- partial-width Stage 5A remains green.

---

## Phase 3 — Add GPT-NeoX checkpoint ingestion

**Priority: P2. Loader lane; outside attention parity unless detection exposes a real semantic backend gap.**

The current failure occurs before Metal execution: no registered checkpoint format matches the target `gpt-neox-20b` checkpoint.

### 3.1 Reproduce and identify the concrete source format

Record the exact checkpoint/config source used by the M5 matrix and inspect:

- config architecture identifiers;
- tensor container;
- tensor naming convention;
- dtype/storage types;
- tied/untied embedding ownership;
- positional encoding configuration;
- layer norm placement;
- QKV packing/layout;
- any parallel-shard metadata.

Do not implement a filename-specific detector.

### 3.2 Add format detection and mapping

Add a registered checkpoint detector based on stable metadata/config semantics and map model tensors/configuration into existing Celeg IR where those semantics already exist.

Keep detection separate from architecture support. A useful intermediate state is:

```text
checkpoint recognized
    -> precise unsupported semantic capability
```

That is better than retaining the generic `no registered checkpoint format matches` error if the container is understood but an architecture feature is not.

### 3.3 Compact loader fixture

Create a tiny/synthetic GPT-NeoX-format fixture or metadata fixture that exercises detector and tensor-name mapping without requiring the 20B checkpoint in CI.

### 3.4 Close any exposed architecture gap explicitly

If successful loading exposes a missing Celeg semantic rather than a Metal-specific problem, create a separate substage and add CPU/reference coverage first. Do not hide an architecture mismatch in the checkpoint adapter.

### Acceptance

- target `gpt-neox-20b` checkpoint is recognized;
- required tensors/configuration map deterministically into Celeg IR;
- the model loads and generates on Metal M5;
- loader behavior has a compact automated fixture;
- malformed/unsupported NeoX variants fail with precise diagnostics rather than matching too broadly.

---

## Phase 4 — Close Metal latent attention for Ling

**Priority: P2. Maps to Attention Parity Stages 9, 11, and 12.**

The current Ling failure is an explicit missing execution feature: factorized latent attention.

Do not implement factorized latent attention as an ordinary-KV approximation.

### 4.1 Revisit the Metal layout/state abstraction prerequisite

Before latent execution, complete the minimum Stage 9 work needed to represent state ownership, strides, dimensions, page/layout identity, and cache addressing without assuming ordinary `[K,V]` BF16 layout everywhere.

The abstraction should make unsupported state kinds impossible to accidentally route into ordinary Metal kernels.

### 4.2 Projected latent path first

Implement Stage 11 projected latent attention before factorized latent if the factorized path depends on the same state/layout machinery.

Build correctness-first production paths with CPU/CUDA as semantic oracles before introducing Metal-specific optimization.

Cover:

- state allocation/ownership;
- token/decode;
- batched prefill;
- relevant projection dimensions and strides;
- snapshot/session continuation if latent state participates in it;
- shared ownership rules if representable by the IR.

### 4.3 Factorized latent path

Implement Stage 12 with explicit factorized dimensions and operations. Reuse shared mathematical helpers only where they are independent of hardware scheduling.

Add differential fixtures that fail if the factorized state is flattened or interpreted as ordinary K/V state.

### 4.4 Ling acceptance model

Use `Ling-3.0-tiny` as the M5 acceptance model for factorized latent execution.

### Acceptance

- projected latent Metal path passes its CPU/CUDA differential suite;
- factorized latent Metal path passes token/decode and prefill differential fixtures;
- state/layout ownership is explicit and capability-gated;
- `Ling-3.0-tiny` generates on Apple M5;
- ordinary LFM attention/cache paths remain green;
- capability flags do not imply INT8, sparse, or other latent combinations not explicitly tested.

---

## Phase 5 — Add MLX 4-bit ingestion and Metal execution

**Priority: P3. New format/quantization lane; outside ordinary GGUF quant support.**

The Qwen 27B failures are currently `unsupported dtype`. Treat MLX 4-bit as its own semantic storage format rather than mapping it by name to GGUF Q4 kernels.

### 5.1 Define the MLX quantization contract

From checkpoint metadata, explicitly model the parameters required to reconstruct weights, including as applicable:

- quantization bit width;
- group size;
- scale dtype/layout;
- zero/bias representation;
- packed value order;
- tensor logical shape vs packed storage shape;
- per-tensor metadata/version differences.

Do not key behavior on the filename `4bit` alone.

### 5.2 Reference dequantization first

Add a simple canonical CPU/reference dequantizer for the exact MLX 4-bit representation(s) used by the target models.

Create tiny deterministic fixtures with known packed bytes/scales and expected float values before implementing Metal kernels.

### 5.3 Loader and internal representation

Extend checkpoint ingestion so MLX quantized tensors preserve the metadata required by execution. Avoid lossy conversion to a generic four-bit tag that cannot distinguish group size/layout.

### 5.4 Metal execution

Implement native Metal dequantized GEMV/GEMM or a staged correctness-first route as appropriate.

If a temporary materialization fallback is used during bring-up, it must be:

- explicit in logs/capabilities;
- covered by correctness tests;
- not described as native MLX-4 Metal execution;
- separated from the final native performance milestone.

Then add native kernels and remove/retain the fallback according to measured usefulness.

### 5.5 Qwen acceptance models

Use both target checkpoints:

- `Qwen3.6-27B-MLX-4bit`;
- `Qwen3.8-27B-4bit`.

Because these are large models, keep real-device acceptance generation short and bounded. Correctness should primarily come from compact quant/dequant and matrix-operation fixtures.

### Acceptance

- both target MLX 4-bit checkpoints load without dtype rejection;
- canonical reference dequantization has deterministic fixtures;
- production Metal math agrees with reference within quantized tolerance;
- both Qwen models generate bounded non-empty output on M5;
- ordinary GGUF Q4/Q5/Q6/Q8 paths do not change semantics as a side effect;
- loader uses semantic metadata rather than filename heuristics.

---

## Phase 6 — Final all-model Metal regression closure

**Priority: release/closure gate. Maps to and extends Attention Parity Stages 14/15.**

### 6.1 Promote differential hardening earlier

Do not wait until the original Stage 14 ordering to build all differential infrastructure. Pull forward only the pieces needed by current work:

- block quant/dequant probes for Q5_K and MLX 4-bit;
- compact GEMV/GEMM differentials;
- RoPE frequency probes;
- first-divergence tensor/logit tracing;
- state-layout probes for latent attention.

Stage 14 still owns broad final hardening; these are enabling slices brought forward because they materially reduce debugging risk.

### 6.2 Final manifest

The same manifest introduced in Phase 0 becomes the final M5 runtime suite. Pin enough identity to make results reproducible:

- stable model/checkpoint identifier;
- relevant config/quant metadata fingerprint;
- fixed prompt;
- generation budget;
- expected capability lane.

Avoid exact long generated strings as the primary regression expectation. Prefer deterministic logits/token ids where the fixture can guarantee them, then bounded non-empty/coherent generation for real-model smoke acceptance.

### 6.3 CI split

Hosted CI:

- compile/link all Metal kernels and test targets;
- execute host semantic tests;
- use compact/synthetic fixtures appropriate to runner limits.

Physical M5 acceptance:

- execute the real Metal tests enabled for device runtime;
- execute the all-model manifest;
- emit JSON + Markdown report;
- retain enough previous result metadata to identify the first regressing commit externally or during bisect.

Do not put multi-gigabyte/27B real-model downloads into ordinary hosted CI unless infrastructure is deliberately provisioned for them.

### Final acceptance matrix

The closure report must show green for:

```text
LFM2.5-350M safetensors
LFM2.5-350M GGUF BF16
LFM2.5-350M GGUF F16
LFM2.5-350M GGUF Q4_0
LFM2.5-350M GGUF Q4_K_M
LFM2.5-350M GGUF Q5_K_M
LFM2.5-350M GGUF Q6_K
LFM2.5-350M GGUF Q8_0
LFM2.5-350M GGUF QAD-Q4_0
LFM2.5-8B-A1B MoE
Qwen3.5-0.8B
gemma-4-E4B-it
Ling-3.0-tiny
gpt-neox-20b
Qwen3.6-27B-MLX-4bit
Qwen3.8-27B-4bit
```

## Mapping to the existing attention roadmap

| All-model lane | Existing roadmap relationship |
| --- | --- |
| Phase 0 harness truth | pulls a focused slice of Stage 14 forward |
| Q5_K_M | correctness defect outside attention parity |
| Gemma 4 scaled RoPE | Stage 5B/5C |
| Qwen3.5 M-RoPE theta | Stage 5C parameterization/generalization |
| GPT-NeoX checkpoint | loader/checkpoint work outside attention parity |
| Ling projected/factorized latent | Stage 9 prerequisite + Stages 11/12 |
| MLX 4-bit | loader/quantization work outside attention parity |
| final M5 matrix | Stage 15 plus real-device model acceptance |

The remaining attention roadmap Stages 6-10 are not deleted or silently reordered by this overlay. BlockSparse, DynamicSparse, INT8 KV, and general layout work remain independent semantic commitments. The latent lane may legitimately pull the minimum necessary Stage 9 layout work forward because it is a dependency of Ling.

## Risk register

### Q5_K_M label vs actual tensor formats

A `_Q5_K_M` filename can describe a quantization recipe containing different concrete storage types by tensor. Always inspect metadata before declaring a Q5_K kernel guilty.

### Dynamic NTK semantic ambiguity

Do not ship a third formula. Resolve host vs CUDA numerically and establish the canonical contract before implementing the Metal mirror.

### GPT-NeoX loader success may reveal a second gap

Recognizing the checkpoint may expose an architecture or tensor-layout feature not currently represented. Keep detector success separate from execution capability.

### Latent attention can leak ordinary-layout assumptions

Stage 9 exists to prevent ordinary BF16 KV assumptions from becoming implicit in general state execution. Do not bypass it with Ling-specific pointer arithmetic.

### MLX is not GGUF

Similar bit width does not imply the same packing, scaling, grouping, metadata, or matrix-kernel contract.

### Long generated text is a weak numerical oracle

Use reference tensors/logits/token ids for deterministic compact fixtures; use real-model text as bounded end-to-end acceptance.

### Large-model runtime cost

The 27B acceptance rows should use short prompts/generation budgets and must not turn every developer build into a large model benchmark.

## Commit discipline

Keep fixes independently bisectable:

1. harness/reporting;
2. Q5_K semantic fixture;
3. Q5_K production fix;
4. RoPE canonical semantics;
5. Metal RoPE mirror and production migration;
6. M-RoPE theta generalization;
7. GPT-NeoX loader;
8. latent layout/runtime slices;
9. MLX reference/loader;
10. MLX Metal kernels;
11. final matrix/CI documentation.

Do not combine a model adapter, a new kernel, and an unrelated backend optimization in one commit merely because the same real model exercises all three.

## Immediate next actions

1. Implement Phase 0's smoke result classifier enough that Q5_K_M cannot be green with zero generated tokens.
2. Dump the failing Q5_K_M file's actual tensor quantization metadata.
3. Add the block-level Q5_K CPU/reference-vs-Metal differential harness.
4. Fix the first proven Q5_K semantic/dispatch defect and rerun every LFM canary.
5. Resume Stage 5B by canonicalizing all `RopeScalingSpec` variants and resolving Dynamic NTK host/CUDA disagreement.
6. Use Gemma 4 and Qwen3.5 as the two real-device Stage 5 acceptance models.

The plan should be updated from observed evidence. A real-model failure can reprioritize a lane, but capability truth and compact differential tests remain the gate for declaring it closed.
