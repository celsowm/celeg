# Support unsloth/Qwen3.8-27B-NVFP4 on CUDA: qwen3_5 linear attention + FP8 + NVFP4

Status: **Not started.** Planned 2026-08-20, revised 2026-08-20 after verifying every claim against
the live checkpoint (`config.json`, `model.safetensors.index.json`) and the current tree. This is a
multi-phase, multi-session effort — see Sequencing. Update the phase checklist as work lands.

- [x] Phase 1 — per-tensor quant-format infrastructure (pure refactor)
- [x] Phase 2 — `linear_attn` binding + vision/MTP fit, running in bf16
- [ ] Phase 3 — FP8 W8A8 kernel
- [ ] Phase 4 — NVFP4 W4A4 kernel
- [ ] Phase 5 — wire real `quantization_config` into the per-tensor resolver
- [ ] Phase 6 — end-to-end verification against the real checkpoint + docs

> **Naming, not a typo.** The repo is `unsloth/Qwen3.8-27B-NVFP4` (base `Qwen/Qwen3.8-27B`), but its
> `model_type` is `qwen3_5` and its architecture class is `Qwen3_5ForConditionalGeneration` — the
> Qwen3.8 releases ship on the `qwen3_5` modelling code. Both spellings below are deliberate: the
> repo id is `Qwen3.8`, the architecture is `qwen3_5`.

## Context

The goal is `unsloth/Qwen3.8-27B-NVFP4` on CUDA with real 4-bit execution (not a bf16 upconvert — a
27B model at bf16 needs ~54GB, more than the 32GB RTX 5090 available here). The checkpoint is 23.4GB
total, no MoE experts, and fits the card comfortably once quantized.

### What the checkpoint actually contains (verified, not inferred)

Text config: `hidden_size` 5120, 64 layers, `intermediate_size` 17408, vocab 248320, `head_dim` 256,
24 query heads / 4 KV heads, `rms_norm_eps` 1e-6, untied embeddings.

- **Hybrid layer stack.** `layer_types` alternates `linear_attention` ×3 then `full_attention`
  (`full_attention_interval: 4`), 64 layers total → 16 full-attention layers at indices 3, 7, …, 63.
- **Full-attention layers** carry `attn_output_gate: true`, `partial_rotary_factor: 0.25`, q/k norms,
  and interleaved mRoPE (`mrope_section: [11, 11, 10]`, `rope_theta` 1e7).
- **`linear_attn` is a full gated-DeltaNet, not a 3-tensor module.** Per linear-attention layer the
  index lists nine tensors: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d.weight`,
  `dt_bias`, `A_log`, `norm.weight`, `out_proj`. Config gives `linear_num_key_heads` 16,
  `linear_key_head_dim` 128, `linear_num_value_heads` 48, `linear_value_head_dim` 128,
  `linear_conv_kernel_dim` 4, `output_gate_type: "swish"`. Expected (to confirm from the safetensors
  header, the index carries no shapes): `in_proj_qkv` → 16·128 + 16·128 + 48·128 = 10240 rows,
  `in_proj_z` → 6144, `in_proj_a`/`in_proj_b` → 48 each, `conv1d` → fused q/k/v depthwise, width
  10240, kernel 4.
- **Two coexisting quant formats.** `quantization_config.format: "mixed-precision"` with two groups:
  - `group_0`, `format: "float-quantized"` — FP8 e4m3, per-**channel** static weight scales
    (`weight` + `weight_scale`) with **dynamic per-token** FP8 activation quant (W8A8). Targets:
    `re:.*self_attn\.(q|k|v|o)_proj$`, `re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$`,
    `re:.*lm_head`, `re:.*layers\.(56|57|…|63)\.mlp\.(gate|up|down)_proj$`. Note the `linear_attn`
    projections are FP8 too — an earlier draft of this plan missed that.
  - `group_1`, `format: "nvfp4-pack-quantized"` — NVFP4 (`weight_packed` + per-16-block fp8
    `weight_scale` + per-tensor fp32 `weight_global_scale`), plus `input_global_scale` for
    `dynamic: "local"`, group-16 activation quant: genuine W4A4, both operands quantized. Target:
    `re:.*mlp\.(gate|up|down)_proj$` — i.e. every MLP **except** the layers 56–63 claimed by
    `group_0`, which is a plain first-match-wins ordering question, not a special case.
  - `ignore` lists every `model.visual.blocks.*` linear → the vision tower stays bf16.
- **Not covered by either group:** all `mtp.*` tensors (`mtp.fc.weight`, `mtp.norm.weight`,
  `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, and
  `mtp.layers.0.{self_attn.*,mlp.*,*_layernorm}`) load as bf16. `in_proj_a`/`in_proj_b`/`conv1d`/
  `A_log`/`dt_bias`/norms are likewise unquantized.
- **`k_scale`/`v_scale` scalars** sit next to each full-attention `self_attn` (fp8 KV-cache
  calibration). They are safe to ignore while the KV cache stays bf16; the loader must not choke on
  unbound tensors.

### What celeg already has (verified in-tree)

- **`qwen3_5` already resolves through the generic path.** `tests/architecture_resolution_test.cpp:299`
  asserts `qwen3_5` selects the `automatic` architecture, and `tests/cpu_mrope_test.cpp` builds and
  runs a real `model_type: qwen3_5` checkpoint end-to-end on CPU — `model.language_model.*` tensor
  naming, nested `text_config`, interleaved mRoPE with `mrope_section`, `layer_types`. No new
  architecture descriptor is expected to be needed; treat adding one as a last resort.
- **Per-layer mixed mixers already work** (`src/model/descriptor/architecture.cpp` reads HF
  `layer_types` into a `LayerSpec.mixer` variant per layer). Qwen3.5's 3:1 hybrid needs no new
  dispatch mechanism.
- **Gated DeltaNet is already implemented**, in two name dialects
  (`src/model/inference/rules_recurrent.cpp`): `FusedGatedDeltaRule` for GGUF `blk.N.{attn_qkv,
  attn_gate,ssm_alpha,ssm_beta,ssm_conv1d,ssm_dt.bias,ssm_a,ssm_norm,ssm_out}`, and
  `FactorizedGatedDeltaRule` for HF `model.layers.N.attention.{q,k,v,f,b,g}_proj` + per-stream
  conv1d. Qwen3.5's `linear_attn` maps **one-to-one onto the fused dialect** — `in_proj_qkv`↔
  `attn_qkv`, `in_proj_z`↔`attn_gate`, `in_proj_a`↔`ssm_alpha`, `in_proj_b`↔`ssm_beta`, `conv1d`↔
  `ssm_conv1d`, `dt_bias`↔`ssm_dt.bias`, `A_log`↔`ssm_a`, `norm`↔`ssm_norm`, `out_proj`↔`ssm_out`.
  This is a third name dialect over existing semantics, **not a new mixer or new kernel** — the
  single largest de-risking correction to the original plan.
- **A Qwen-style ViT already exists and already binds these exact tensor names.**
  `src/runtime/vision/safetensor_projection.cpp` implements `model.visual.{patch_embed.proj,
  pos_embed,blocks.N.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2},merger.*}`.
  (`patch_projection.cpp` is the *other*, LFM2 `model.vision_tower.*` path — not the one to extend.)
  Its hardcoded constants match this checkpoint's `vision_config` exactly: 27 blocks, hidden 1152,
  16 heads × 72, patch 16, temporal 2, 48×48 position grid. **One mismatch:** the merger output width
  is hardcoded to 2048 (`linear(..., 2048)`, `VisualEmbedding result{2048, …}`) while this model's
  `out_hidden_size` is 5120. That constant must be derived from the `merger.linear_fc2` shape.
- **MTP exists** (`src/backend/cuda/model/mtp_execution.cu`), CUDA-only, constrained to "exactly one
  auxiliary full-attention decoder layer". Qwen3.5 has exactly one such layer, but also `mtp.fc`,
  `mtp.norm`, and two `pre_fc_norm_*` tensors — fit is plausible, unverified.
- **Weight storage is already per-tensor** (`LinearWeight::storage` is a variant of
  `Bf16LinearStorage`/`Int8LinearStorage`/`Int4LinearStorage`/`GgufLinearStorage`, mixed today for
  GGUF-native models). The blockers to mixing FP8 and NVFP4 in one model are that **kernel selection
  is global** (`src/backend/cuda/runtime/gemm_dispatcher.cpp:202` — `binding.kernel =
  plan.linear_kernel()`, one value per model, set in `src/backend/cuda/execution_plan.cpp:80-90`) and
  the **loader's `weight_mode_` is a single global field**
  (`src/backend/cuda/model/linear_loader.cpp`, ~15 call sites).
- **No FP8 support at all** (`CUDA_R_8F_E4M3`/e4m3: zero hits under `src/backend/cuda/`). **One
  dynamic-activation-quant precedent**: `launch_quantize_q8_1`
  (`src/backend/cuda/kernels/mmq.cu:636`) quantizes bf16 activations to int8 before the GGUF MMQ
  kernel — the pattern to imitate for both FP8 and NVFP4 activation quant. The fp8/fp4 kernels
  themselves are net-new.

## Standing constraint: stay architecture-agnostic

celeg's established rule (see `docs/inference_report.md`: "implemented in the
automatic/architecture-agnostic resolution path ... not as per-model configuration... an earlier
draft ... used [a per-model descriptor] and was deliberately replaced with this generic version
instead, since a JSON file per model family doesn't scale") applies to every phase here:

- Phase 2's `linear_attn` binding must be a **name dialect** on the existing gated-DeltaNet rule
  (probe on the tensor grammar, exactly as the two existing dialects do), not a `qwen3_5` branch.
  Likewise the ViT's merger width must come from the tensor shape, not from a model check.
- Phase 5's `quantization_config` parser must interpret `config_groups[*].format` + `targets`
  generically — any compressed-tensors-style checkpoint should resolve correctly. Derive
  quantized-vs-not per tensor by matching the `targets` regexes against the tensor's own name; never
  hardcode "layers 56–63".
- If a generic rule and a per-model shortcut both solve a step, prefer the generic rule, even if it
  takes more work — this has already paid off once here (the Nanbeige fix cited above) and is a hard
  constraint, not a style preference.

## Sequencing

Each phase is built and verified independently; a failure in one is otherwise very hard to isolate
from the others. Dependencies flow downward.

**Phase 1 — Per-tensor quant-format infrastructure (foundation for 3 & 4).**
Add `LinearKernelKind` as a field on `LinearWeight` (loader-set per tensor, mirroring how the loader
already knows which storage variant it populated) instead of solely on `CudaExecutionPlan`. Change
`GemmDispatcher::compile_linear_binding` (`gemm_dispatcher.cpp:190`) to take `binding.kernel` from
the weight, falling back to `plan.linear_kernel()` when the weight sets none — every existing
single-format model keeps working unchanged. Replace `WeightLoader`'s global `weight_mode_`
(`src/backend/cuda/model/linear_loader.cpp`) with a per-tensor-name resolver; for this phase the
resolver just returns the existing global mode, making it a pure refactor with no behavior change.
Verified by the full `ctest` suite staying green (89/89) with zero new formats.

**Phase 2 — Architecture fit, running in bf16 first (no quantization yet). DONE.**
Verified against the real checkpoint's safetensors header (range-fetched, not downloaded) and landed
against synthetic checkpoints (`automatic_inference_test.cpp`, `layer_inference_rule_test.cpp`,
`qwen35_vision_test.cpp`). What actually shipped, including two gaps the original plan missed
entirely (not just "untested combinations" — these code paths did not exist):
  - **New `linear_attn` name dialect** (`LinearAttnGatedDeltaRule` in `rules_recurrent.cpp`),
    structurally identical to `FusedGatedDeltaRule` (same qkv/conv-width formula, same per-role
    shapes) — confirmed against the real safetensors header: `in_proj_qkv` `[10240,5120]`,
    `in_proj_z` `[6144,5120]`, `in_proj_a`/`in_proj_b` `[48,5120]`, `conv1d` `[10240,1,4]`, `norm`
    `[128]`, `out_proj` `[5120,6144]` — all match the `key_heads=16, key_dim=128, value_heads=48,
    value_dim=128` formula exactly. Shipped as a fourth full rule class (matching the existing
    three-class precedent in the file) rather than a shared alias table — the existing dialects
    aren't table-driven either, so a table would have been a bigger, unprecedented refactor for no
    behavioral gain. New `GatedDeltaFacts::linear_{key,value}_heads/{key,value}_dim/conv_kernel`
    fields (separate from the factorized dialect's `key_heads`/`value_heads`) avoid `aliases()`
    treating Qwen3.5's distinct key vs. value head counts as conflicting metadata for one fact.
  - **Two real gaps in the generic/automatic path, not just "untested":** `partial_rotary_factor`
    (Qwen3.5's actual config key) was never aliased to `rotary_fraction` — only the differently-named
    `rotary_fraction` key was. And M-RoPE sectioning (`mrope_section`/`mrope_interleaved`) was *only*
    wired in the per-model descriptor path (`src/model/descriptor/architecture.cpp`); the generic
    `automatic` architecture that Qwen3.5 actually resolves through never constructed a
    `MultiAxisRopeSpec` at all. `cpu_mrope_test.cpp`'s green baseline did not prove otherwise — it
    only checks chunked-vs-scalar prefill self-consistency, not that mRoPE math is applied. Fixed
    generically: `InferredRopePosition` gained `mrope_sections`/`mrope_interleaved`, `metadata.cpp`
    aliases `partial_rotary_factor` and `mrope_section(s)`/`mrope_interleaved` (with the existing
    `text_config.` fallback covering the nested-config case for free), and `rules_attention.cpp`
    builds a `MultiAxisRopeSpec` when sections are present. New synthetic test in
    `automatic_inference_test.cpp` proves the full combination (gated output, 0.5 rotary fraction,
    interleaved 3-axis sections) resolves correctly through `ArchitectureCatalog::select` with no
    descriptor JSON registered.
  - **Vision merger width fixed**: `safetensor_projection.cpp` now derives it from
    `merger.linear_fc2`'s own shape (`merger_out_`) instead of the hardcoded 2048; `qwen35_vision_test`
    assertions are shape-derived (`width > 0`, consistent across calls) instead of hardcoded to 2048.
  - **MTP tensor-name fit confirmed** by inspecting the real checkpoint's `model_mtp.safetensors`
    shard: `mtp.layers.0.{input_layernorm,post_attention_layernorm,self_attn.{q,k,v,o}_proj,
    self_attn.{q,k}_norm,mlp.{gate,up,down}_proj}` plus `mtp.fc`/`mtp.norm`/`mtp.pre_fc_norm_*` is a
    dense (non-MoE) MTP layer that matches `mtp_weight_setup.cpp`'s existing dense-MLP MTP path
    tensor-for-tensor. No code change needed; true end-to-end exercise still requires the real
    checkpoint (Phase 6).
  - **Unbound tensors are already tolerated**: the loader only ever requests specific known tensor
    names from the repository — there is no "every tensor must be consumed" completeness check
    anywhere in the codebase — so `k_scale`/`v_scale` and (later) the FP8/NVFP4 scale/packed tensors
    simply go unread without error. No change needed.
  - **New, real blocker for a *real-checkpoint* bf16 baseline, not previously identified:** the
    checkpoint has no bf16 copies of `linear_attn`'s `in_proj_qkv`/`in_proj_z`/`out_proj`, every
    `self_attn` q/k/v/o, or most `mlp` gate/up/down — they are natively `F8_E4M3` or NVFP4-packed on
    disk, and celeg's `TensorDType` enum (`include/celeg/checkpoint/tensor.hpp`) has no FP8/NVFP4
    variant at all today. So "ignore `quantization_config`, load bf16" is not actually possible
    against the real checkpoint yet — it requires at minimum a dequantize-on-load path (read the raw
    e4m3/nvfp4 bytes + scale tensors, upconvert to bf16 at load time), which is real work adjacent to
    but distinct from Phase 3/4's actual GEMM kernels. The vision tower is the one part of the graph
    that *is* genuinely bf16-native on disk (every `model.visual.*` linear is in `ignore`), so
    `qwen35_vision_test` against `CELEG_QWEN35_MODEL` should work once the checkpoint is downloaded;
    full text-generation verification against the real checkpoint is deferred until Phase 3/5 give the
    loader a way to read the on-disk dtypes at all. Revise Phase 3/5 to include this dequant-on-load
    step explicitly rather than assuming it falls out of the GEMM kernel work for free.
  - Did not add an architecture descriptor JSON (`src/model/descriptor/`) — the generic inference
    rules now express this config shape fully, including the mRoPE gap that was closed above.

**Phase 3 — FP8 W8A8 kernel.**
New `Fp8LinearStorage` (per-channel fp32 scale + e4m3 packed weight), new
`LinearKernelKind::Fp8W8A8`, a dynamic **per-token** activation-quantization kernel (modeled on
`launch_quantize_q8_1`) producing e4m3 activations, and an fp8 matmul with a manual per-channel/
per-token scale post-multiply (see spike findings below — not cuBLASLt's native scale-vector mode).

*Standalone cuBLASLt spike, done (throwaway scratchpad program at
`/tmp/.../scratchpad/fp8spike/`, not committed — celeg had zero existing fp8 cuBLASLt usage to copy
from). Findings, on this machine's RTX 5090 (sm_120) / CUDA 13.2:*
  - **Plain (unscaled) `CUDA_R_8F_E4M3 × CUDA_R_8F_E4M3 → fp32`/`bf16` matmul works and is exact**:
    ran a 4×8×32 GEMM through `cublasLtMatmul` with the same TRANSA=T/TRANSB=N layout convention
    `gemm_dispatcher.cpp`'s bf16 path already uses, compared against a scalar dot-product reference on
    the same quantized e4m3 values — bit-exact (`max_abs_err=0`). The core fp8 tensor-core path is
    real and usable here.
  - **cuBLASLt's native per-channel/per-token scaling
    (`CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` on `CUBLASLT_MATMUL_DESC_A_SCALE_MODE`/
    `B_SCALE_MODE`) is *not* supported on this GPU/toolkit combination** — `cublasLtMatmulAlgoGetHeuristic`
    returns `CUBLAS_STATUS_NOT_SUPPORTED` for every variant tried (fp32 and bf16 output, K=32 and
    K=128, both operands using outer-vec scale). This is despite the CUDA 13.2 header declaring the
    mode and it being exactly what our checkpoint's per-channel-weight + per-token-activation W8A8
    scheme needs semantically. Consumer Blackwell (RTX 50-series) most likely lacks this specific
    kernel in cuBLAS's selection tables even though the fp8 tensor cores themselves work fine for the
    unscaled case — a library/SKU gap, not a math or API-usage error (confirmed by testing several
    transpose/dtype/K-size combinations, not just one).
  - **Conclusion — do not depend on cuBLASLt for the scale application.** Run the raw fp8×fp8→fp32
    matmul unscaled (as validated above), then apply `y[m,n] *= act_scale[m] * weight_scale[n]`
    (+ bias) as a separate elementwise kernel. This is exactly the pattern
    `launch_w8a16_linear` (`src/backend/cuda/kernels/linear.cuh`) already uses for the existing
    W8A16 int8-weight path (`accum * row_scale` inside a hand-written kernel) — Phase 3's W8A8 kernel
    should follow that same in-house convention (custom kernel with the scale multiply built in, not a
    cuBLASLt scale-mode attribute) for both the matmul-epilogue scale and, if throughput matters more
    than the cuBLASLt call overhead at small M, potentially the whole GEMM. No W8A8 kernel of any kind
    exists yet (confirmed via grep) — this is genuinely net-new, not an extension of `launch_w8a16_linear`.
  - Still open before writing the production kernel: confirm the scale-application math against a
    scalar reference matching the checkpoint's actual per-channel weight-scale shape (`[out_features,1]`)
    and per-token dynamic activation quant (not yet spiked — only the raw unscaled matmul and the
    scale-mode-unsupported finding are validated so far).

*Kernel + dispatch wiring, done.* Landed (not just spiked) the full path, committed:
  - `launch_quantize_e4m3_per_row` (`src/backend/cuda/kernels/linear.cuh`): one kernel serves both
    per-token dynamic activation quant and per-channel weight quant (identical "row absmax -> e4m3"
    math; only *when* it runs differs — per forward pass for activations, once at load time for
    weights). Block-level reduction (warp shuffle + shared-memory cross-warp reduce), any row length.
  - `launch_fp8_scale_apply`: the manual outer-product dequant epilogue the spike's findings called
    for (`y[m,n] = raw[m,n] * act_scale[m] * weight_scale[n]`, with the existing `beta`-accumulate
    convention).
  - `Fp8LinearStorage` (e4m3 data + per-row fp32 scales, same shape convention as `Int8LinearStorage`)
    added to the `LinearStorage` variant, and `LinearKernelKind::Fp8W8A8` added to the enum --
    including the `slice_rows()` visitor branch and the `execution_plan.cpp` name-printer switch that
    would otherwise silently miss it. Unused by any loader yet (Phase 5), same safe staging as Phase 1.
  - `GemmDispatcher::linear_fp8_w8a8` + `get_or_create_fp8_lt_plan` wire it into the real dispatch
    switch (`GemmDispatcher::linear`'s `case LinearKernelKind::Fp8W8A8`), reusing the raw (unscaled)
    matmul plan shape validated by the spike.
  - New test in `tests/cuda_kernels_test.cu` exercises the *whole wired path* through
    `GemmDispatcher::linear()` (not just the raw kernels in isolation) against a reference built from
    the kernel's own quantized values (round-tripped through the same quantize kernel under test, not
    an independently reimplemented e4m3 rounding rule) — passes at 2% relative tolerance.
  - **New finding: fp8 cuBLASLt heuristics need aligned shapes.** `n=3` (an arbitrary small test size)
    returned no available algorithm (`cublasLtMatmulAlgoGetHeuristic` succeeds but returns zero
    results) while `n=8` works; real Qwen3.5 tensor widths (5120, 6144, 10240, 17408, head_dim
    multiples of 128) are almost certainly always aligned in practice, but this needs an explicit
    guard or bf16 fallback for the general case before Phase 5 wires real checkpoints through this
    path — don't assume every shape works just because the common ones do.

**Phase 4 — NVFP4 W4A4 kernel.**
`Nvfp4LinearStorage` (packed e2m1 + per-16-block `UE4M3` scale + per-tensor fp32 global scale),
`LinearKernelKind::Nvfp4W4A4`, a dynamic per-16-block activation-quantization kernel to NVFP4 (using
the checkpoint's static `input_global_scale` as calibration), and a cuBLASLt block-scaled matmul with
`CUDA_R_4F_E2M1` operands and `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` scale mode (available: CUDA
13.2, RTX 5090/sm_120). **Spike this standalone first** to pin down exact descriptor attributes,
K-dimension alignment/padding requirements, and confirm the block-scaled API behaves as documented
for this shape — resolve empirically, don't guess. Validate against a scalar e2m1-LUT reference.
Note the `weights.actorder: "static"` flag in `group_1`; confirm whether it implies a permutation
that must be undone at load time.

**Phase 5 — Wire the real `quantization_config` into the Phase-1 per-tensor resolver.**
Parse `config_groups` (`format: "float-quantized"` → FP8, `"nvfp4-pack-quantized"` → NVFP4) plus the
`ignore` list into the Phase-1 resolver, matching each group's `targets` regexes against tensor names
with first-match-wins so the layers-56–63 overlap resolves the way vLLM/compressed-tensors resolves
it. Also honor `weights.strategy` (`channel` vs `tensor_group`) and `input_activations.dynamic`
(`true` per-token vs `"local"` per-group-16) rather than assuming them per format. This is what makes
the loader pick FP8 vs NVFP4 vs bf16 per tensor instead of one global `--weight-mode`.

**Phase 6 — End-to-end verification and docs.**
Full quantized run of the real 27B checkpoint; compare output against the Phase-2 bf16 baseline for
coherence/consistency; add to `scripts/run_model_sweep.py`; write up in `docs/inference_report.md`.

## Tests (per phase, added incrementally — not all upfront)
- Phase 1: existing `ctest` suite stays green, zero behavior change (pure refactor).
- Phase 2 (done): `automatic_inference_test.cpp` gained a synthetic `qwen3_5` checkpoint (one
  linear-attention + one full-attention layer, via `CheckpointView`/`ArchitectureCatalog::select`
  rather than a full CPU run) asserting the `linear_attn` dialect's resolved geometry and the
  full-attention gate+partial-rotary+interleaved-mRoPE combination (`MultiAxisRopeSpec`);
  `layer_inference_rule_test.cpp`'s builtin-rule-count/specificity invariant updated for the new rule;
  `qwen35_vision_test`'s width assertions made shape-derived. MTP fit confirmed by tensor-name
  inspection, not a new test (no code changed). Full `ctest` (89/89) green throughout — non-regression
  confirmed for existing models on both backends.
- Phase 3 & 4: synthetic-weight kernel correctness tests (dequant/matmul vs. scalar reference) per
  format, before wiring into the loader.
- Phase 5: a `quantization_config` parser test against a minimal `config.json` fragment matching the
  real schema, including the deliberate layers-56–63 overlap between the two groups.
- Phase 6: full model run, `classify_output()` coherence/correctness check (reuse from
  `scripts/run_model_sweep.py`), VRAM sanity check (~14–17GB expected).

## Verification (cumulative, end state)
1. `ninja -j32 && ctest -j8` green throughout (89/89 baseline, growing with new tests).
2. `unsloth/Qwen3.8-27B-NVFP4` downloaded via celeg's existing `--repo` HF fetch (878GB free disk,
   23.4GB download) and producing coherent, correct output via
   `celeg-run --repo unsloth/Qwen3.8-27B-NVFP4 --prompt "What is the capital of France?"`.
3. VRAM in the expected ~14–17GB range, confirming real 4-bit/8-bit storage is in effect (not a bf16
   fallback).
4. Non-regression across the existing model sweep on both backends.

## Risks
- **Two new kernel formats at once** (FP8, NVFP4) is why Phase 2 deliberately establishes a bf16-only
  baseline first. Do not skip it. (Architecture risk is now much lower than first assessed — see
  below — so the bf16 phase should be short.)
- **Resolved in Phase 2:** `linear_attn` shapes confirmed against the real safetensors header
  (packing order matches the fused-dialect formula exactly); the output-gate + partial-rotary +
  interleaved-mRoPE combination now resolves correctly (and two real gaps in the generic path were
  found and fixed along the way — see Phase 2 above, `partial_rotary_factor` and `mrope_section` were
  simply never read before); vision merger width is shape-derived; MTP tensor names confirmed to match
  the existing dense-MLP MTP path exactly.
- **New: no dequant-on-load path for FP8/NVFP4-native tensors.** The real checkpoint has no bf16
  copies of most linear weights — they are natively `F8_E4M3` or NVFP4-packed — and
  `include/celeg/checkpoint/tensor.hpp`'s `TensorDType` has no FP8/NVFP4 variant. A true bf16 baseline
  run against the real checkpoint (not just synthetic-checkpoint architecture resolution) needs this
  before it's possible; fold it into Phase 3/5 rather than assuming it's free. Only the vision tower is
  genuinely bf16-native on disk today.
- **cuBLASLt fp8 and NVFP4 block-scaled APIs are both unproven in this codebase** — each phase
  includes a standalone spike specifically to de-risk this before integration.
- **No CUTLASS fallback planned** — if cuBLASLt proves unworkable for either shape, that is a
  materially larger re-scope requiring a check-in with the user, not a silent substitution.
- **Total scope is still large** — realistically multiple sessions; each phase is an independently
  valuable, independently verifiable unit rather than one unreviewable change.
