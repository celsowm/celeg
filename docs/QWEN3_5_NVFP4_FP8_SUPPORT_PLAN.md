# Support unsloth/Qwen3.8-27B-NVFP4 on CUDA: qwen3_5 linear attention + FP8 + NVFP4

Status: **Not started.** Planned 2026-08-20, revised 2026-08-20 after verifying every claim against
the live checkpoint (`config.json`, `model.safetensors.index.json`) and the current tree. This is a
multi-phase, multi-session effort — see Sequencing. Update the phase checklist as work lands.

- [x] Phase 1 — per-tensor quant-format infrastructure (pure refactor)
- [x] Phase 2 — `linear_attn` binding + vision/MTP fit, running in bf16
- [x] Phase 3 — FP8 W8A8 kernel
- [x] Phase 4 — NVFP4 W4A4 kernel (native block-scaled tensor-core path, verified bit-exact; dequant-to-bf16 fallback for unsupported shapes)
- [x] Phase 5 — per-tensor FP8/NVFP4 loading, self-describing (no `quantization_config` regex parser)
- [~] Phase 6 — end-to-end verification against the real checkpoint + docs (checkpoint loads and runs cleanly; several real loader/architecture bugs found and fixed; one confirmed NVFP4 scale-direction bug fixed; generation is measurably closer to sane but not yet coherent -- see status below)

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
- Phase 5's per-tensor FP8/NVFP4 detection must generalize to any compressed-tensors-style checkpoint,
  never hardcode "layers 56–63". Landed as sidecar-tensor autodetection rather than a
  `config_groups[*].targets` regex parser — see Phase 5 below for why that's the more generic of the
  two, not a lesser substitute for it.
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
  - **Finding resolved: fp8 cuBLASLt heuristics need aligned shapes, now guarded.** `n=3` (an
    arbitrary small test size) returned no available algorithm (`cublasLtMatmulAlgoGetHeuristic`
    succeeds but returns zero results) while `n=8` works; real Qwen3.5 tensor widths (5120, 6144,
    10240, 17408, head_dim multiples of 128) are almost certainly always aligned in practice, but
    rather than assume that, `GemmDispatcher::linear_fp8_w8a8` now falls back to
    `launch_fp8_w8a8_naive` (`src/backend/cuda/kernels/linear.cuh`) — a plain one-thread-per-output
    dot-product kernel, correct for any m/n/k — whenever `get_or_create_fp8_lt_plan` can't produce
    an algorithm, instead of throwing. Slow but correct is safer than a runtime crash on an
    unanticipated shape. `tests/cuda_kernels_test.cu`'s FP8 test now runs both a `n=8` (cuBLASLt)
    and `n=3` (naive fallback) shape against the same reference to cover both paths.

**Phase 4 — NVFP4 W4A4 kernel.**

*Standalone cuBLASLt spike, done (throwaway scratchpad program at `/tmp/.../scratchpad/nvfp4spike/`,
not committed).* Findings, on this machine's RTX 5090 (sm_120) / CUDA 13.2:
  - **The `VEC16_UE4M3` block-scale mode is not optional for `CUDA_R_4F_E2M1`.** Unlike fp8 (which has
    a working *unscaled* matmul path), calling `cublasLtMatmulAlgoGetHeuristic` on a raw
    `CUDA_R_4F_E2M1 × CUDA_R_4F_E2M1` matmul with no scale mode set returns `CUBLAS_STATUS_INVALID_VALUE`
    (status 7) with zero results — there is no "unscaled fp4 matmul + manual scale kernel" pivot
    available the way there was for fp8. Block scaling has to go through cuBLASLt's native mechanism
    or not through cuBLASLt at all.
  - **With `A_SCALE_MODE`/`B_SCALE_MODE = VEC16_UE4M3` set and per-16-block UE4M3 scale tensors
    supplied in the "obvious" row-major layout (`scales[row * (k/16) + block]`, one scale per row per
    16-element chunk of the row, matching the header comment "for each 16-element block in the
    innermost dimension"), `cublasLtMatmulAlgoGetHeuristic` *does* find an algorithm and
    `cublasLtMatmul` runs without error** — but the result is wrong: ~28% mean relative error against
    a scalar reference built from the exact same quantized e2m1 values and e4m3-rounded scales. At a
    small/unaligned shape (m=4,n=8,k=32) the failure mode is worse than wrong numbers: all but one
    output element come back as exact `0.0`, which is a silent-corruption failure mode, not a clean
    refusal like fp8's small-n case.
  - Ruled out as the cause (both verified independently before concluding it's the scale layout):
    e2m1 nibble pack order / encode-decode rounding (re-derived using CUDA's own
    `__nv_cvt_float_to_fp4`/`__nv_cvt_fp4_to_halfraw` intrinsics instead of a hand-rolled LUT — same
    result), and A/B scale-pointer assignment (swapping A_SCALE_POINTER/B_SCALE_POINTER made the error
    far worse, confirming the original assignment was the right direction, not a fix).
  - **Conclusion: `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`'s scale-factor tensor almost certainly
    requires a specific physical (likely tile-swizzled) layout that isn't documented in the local CUDA
    13.2 headers** (the header says only "see documentation for layout information" — no such doc is
    present locally, and this is a known industry pain point: CUTLASS's own NVFP4 block-scaled GEMM
    recipes compute this layout via a dedicated `Sm1xxBlockScaledConfig`-style helper rather than a
    plain row-major array). Guessing at the swizzle risks a kernel that runs and passes a coarse test
    but is subtly wrong on real checkpoint weights — not acceptable for a correctness-critical GEMM.

*Tried and ruled out: one candidate scale swizzle (128×8 tile / 32×4 sub-interleave).* An LLM
(Gemini) proposed a specific physical layout for the `VEC16_UE4M3` scale tensor — 128-row × 8-`k/16`-
block tiles of 1024 bytes each, with `r_outer*256 + c_outer*128 + r_inner*4 + c_inner` sub-tile
interleaving (`r_outer=row/32`, `c_outer=col/4`), padding regions filled with UE4M3 `0x38` (=1.0)
rather than zero. Implemented it in the spike and tested it apples-to-apples against the naive
row-major layout at a fully tile-aligned m=n=128,k=256 shape (no padding involved either way):
**naive row-major gave 29.4% mean relative error, this swizzle gave 34.4% — worse, not better.**
Not the right formula — but not a dead end either, see below.

*Resolved: found the authoritative layout, verified bit-exact.* Web search turned up NVIDIA's own
cuDNN-frontend docs page, ["The 128×4 Tiled Layout for Block Scaling
Factors"](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/) — a real,
citable NVIDIA source (not another LLM guess), explicitly documented as applying to both MXFP8
(block=32) and NVFP4 (block=16). The layout: 128-row × 4-scale-column tiles of 512 bytes each,
within-tile offset `(row%32)*16 + (row/32)*4 + col`, tiles arranged row-major, rows padded to a
multiple of 128 and scale-columns padded to a multiple of 4 with **zero**-fill (not the earlier
guess's 1.0-fill). Implemented this in the spike and tested against the scalar reference at three
shapes — m=n=128,k=256 (fully aligned, no padding), m=8,n=128,k=256 (m padded), and the original
m=4,n=8,k=32 that previously silently zeroed most outputs — **all three came back bit-exact
(`max_abs_err=0.000000`)**. The earlier "silently zeros most outputs at small shapes" failure mode
was purely an artifact of the wrong scale layout, not an inherent small-shape limitation of the
native path.

*Kernel + dispatch wiring, done — the native block-scaled tensor-core path, not just a fallback.*
With the layout verified, `GemmDispatcher::linear_nvfp4_w4a4` now calls cuBLASLt's native
`CUDA_R_4F_E2M1` + `VEC16_UE4M3` matmul as the primary path:
  - `Nvfp4LinearStorage` (`include/celeg/detail/model/linear_weights.hpp`): packed e2m1 (2 values/byte)
    + one `__nv_fp8_e4m3`-typed UE4M3 scale per `kNvfp4BlockSize`(=16)-element block, row-major, + one
    per-tensor fp32 global scale (`dequant = e2m1_value * block_scale * global_scale`, matching
    compressed-tensors' two-level NVFP4 scaling). Added to the `LinearStorage` variant with a
    `slice_rows()` branch.
  - `launch_quantize_e2m1_per_block` (`src/backend/cuda/kernels/linear.cuh`): dynamic per-16-block
    activation quantization to e2m1 + UE4M3 scale, one thread per block (blocks never overlap, so no
    cross-thread races on the packed output bytes).
  - `launch_swizzle_nvfp4_scale`: rearranges a row-major UE4M3 scale tensor into the verified 128×4
    tiled layout — used on both the activation's scale (computed fresh each call) and the weight's
    scale (`Nvfp4LinearStorage::block_scales`, re-swizzled each call since there's no persistent
    per-weight cache yet — Phase 5 loader work could swizzle once at load time instead).
  - `get_or_create_nvfp4_lt_plan`: mirrors `get_or_create_fp8_lt_plan`'s caching/heuristic pattern,
    but with a real gotcha found by comparing a standalone reproduction against the in-dispatcher
    code path when the first version of this landed with the *right* scale layout but the *wrong*
    plumbing: **`cublasLtMatmulAlgoGetHeuristic` requires the scale POINTER attributes
    (`A_SCALE_POINTER`/`B_SCALE_POINTER`) to already be set on the descriptor, not just the scale MODE
    — omitting them makes the heuristic call fail with `CUBLAS_STATUS_INVALID_VALUE` and zero
    algorithms, even though the mode is set correctly.** This is undocumented (not mentioned in the
    header) and isn't needed for the analogous fp8 path (which doesn't use scale-vector attributes for
    the reasons in the Phase 3 section above). Fixed by setting the pointers (to the current workspace
    buffer addresses) during plan creation as well as before every `cublasLtMatmul` call.
  - `launch_nvfp4_global_scale_apply`: post-multiplies the raw matmul output by
    `weight_global_scale * activation_global_scale` — cuBLASLt's block-scale mode only bakes in the
    per-16-block UE4M3 scale, not the per-tensor global scale, so that has to be applied separately.
  - Falls back to the previous dequant-to-bf16 approach (`launch_dequant_nvfp4`, weight only, activation
    stays exact bf16) only when `get_or_create_nvfp4_lt_plan` finds no algorithm for a given shape —
    same role `launch_fp8_w8a8_naive` plays for the fp8 path, though now-untested whether any real shape
    actually needs it (all three spike-verified shapes found an algorithm).
  - New test in `tests/cuda_kernels_test.cu` quantizes both weight and activation with the actual
    production kernel (`launch_quantize_e2m1_per_block`), builds the scalar reference from those same
    quantized values, then runs the full `GemmDispatcher::linear()` path with
    `LinearKernelKind::Nvfp4W4A4` — for two shapes (m=4,n=8,k=32 and m=8,n=128,k=256) — within 2%
    relative tolerance.
  - Activations don't yet carry a checkpoint-calibrated `input_global_scale` (that's Phase 5 loader
    work); `GemmDispatcher` hardcodes `1.0` for it until then.
  - Note the `weights.actorder: "static"` flag in `group_1` of the real checkpoint's
    `quantization_config`; still unconfirmed whether it implies a permutation that must be undone at
    load time (Phase 5 concern, not resolved here).

**Phase 5 — Per-tensor FP8/NVFP4 loading, done.**
Landed as **self-describing autodetection from the checkpoint's own tensor layout**, not a
`quantization_config`/`config_groups` regex parser — a deliberate design change from the original
plan, reasoned through at the start of this phase rather than assumed:
  - Every quantized-vs-not, and FP8-vs-NVFP4, decision the checkpoint needs is already fully implied
    by which sidecar tensors physically exist for a given weight name: a plain `<name>` tensor of
    dtype `F8_E4M3` plus `<name>_scale` means FP8; `<name>_packed` + `<name>_scale` (dtype `F8_E4M3`)
    + `<name>_global_scale` means NVFP4; neither means dense (bf16/f16/f32, handled unchanged by the
    existing path). This resolves the layers-56–63 group_0/group_1 overlap correctly with zero
    knowledge of "layers 56–63" anywhere in the code, because the two formats are told apart by what's
    actually on disk for that exact tensor, not by a name pattern matched against a JSON schema copy
    of the same information. It generalizes to any compressed-tensors-style checkpoint that mixes
    formats per tensor, including ones this plan never saw.
  - This is the same convention `has_packed_int8_matrix` (`src/checkpoint/packed/int8.cpp`) already
    used for a different checkpoint format, pre-dating this plan — Phase 5 extends that precedent
    rather than introducing a second, JSON-driven mechanism alongside it.
  - `weights.strategy`/`input_activations.dynamic` don't need to be read from JSON either: they're
    exactly what the Phase 3/4 kernels already do unconditionally per storage format (FP8 → per-token
    dynamic e4m3 activation quant; NVFP4 → per-16-block dynamic e2m1 activation quant with the
    checkpoint's calibrated `input_global_scale`) — there is no second strategy either kernel would
    need to switch on.
  - `CheckpointMetadata::from_json` (`src/checkpoint/metadata.cpp`) would have thrown on this
    checkpoint's `config.json` before any weight loading even started: `quantization_config` nests
    `config_groups`, an array of *objects*, which `flatten_json`'s scalar/vector-only scheme can't
    represent (`"unsupported metadata array: config_groups"`). Found by reading the code, not by
    running against the real checkpoint (not downloaded yet). Fixed by skipping that one top-level key
    during flattening — nothing currently reads it, matching the autodetection design above.
  - New: `TensorDType::F8_E4M3` and `::U8` (`include/celeg/checkpoint/tensor.hpp`,
    `src/checkpoint/formats/safetensors.cpp`); `celeg/checkpoint/packed/fp8.hpp`+`.cpp` and
    `.../nvfp4.hpp`+`.cpp` (`has_packed_fp8_matrix`/`load_packed_fp8_matrix`,
    `has_packed_nvfp4_matrix`/`load_packed_nvfp4_matrix`), unit-tested against a synthetic in-memory
    repository (`tests/packed/{fp8,nvfp4}_test.cpp`) the same way `packed_int8_test`/`packed_int4_test`
    already are. `Nvfp4LinearStorage` gained `input_global_scale` (default `1.0`, matching the
    checkpoint's field of the same name when a `<module>.input_global_scale` sidecar is present);
    `GemmDispatcher::linear_nvfp4_w4a4` now multiplies by `weight.input_global_scale` instead of the
    hardcoded `1.0` placeholder from Phase 4.
  - `WeightLoader::load_linear_weight` (`src/backend/cuda/model/linear_loader.cpp`) checks
    `has_packed_fp8_matrix`/`has_packed_nvfp4_matrix` right alongside the existing
    `has_packed_int8_matrix`/`has_packed_int4_matrix` checks, before falling through to the dense
    tensor path, and sets `weight.linear.kernel` to `Fp8W8A8`/`Nvfp4W4A4` — the per-tensor override
    Phase 1 built for exactly this. `load_concat_linear_weight` was deliberately left untouched: its
    only quantization-config-targeted callers are MoE shared-expert fusion, and this checkpoint has no
    MoE experts (confirmed in Phase 2), so it isn't exercised by the real checkpoint.
  - **Still open, and correctly Phase 6's job, not Phase 5's:** the exact on-disk safetensors dtype
    spellings (`"F8_E4M3"` vs e.g. `"F8_E4M3FN"`, `"U8"` for `weight_packed`) and sidecar tensor names
    (`weight_scale`, `weight_global_scale`, `input_global_scale`) are taken from the Phase 2
    quantization_config summary quoted at the top of this doc, not re-verified byte-for-byte against
    the checkpoint here — it still hasn't been downloaded. Same for the `weights.actorder: "static"`
    flag noted in Phase 4: still unconfirmed whether it implies a column permutation that must be
    undone at load time, or is a no-op for this weight/activation strategy combination. If the real
    checkpoint's dtype string or sidecar names differ, `has_packed_fp8_matrix`/`has_packed_nvfp4_matrix`
    simply won't detect it and the tensor falls through to the dense path, which throws a clear
    "unexpected linear tensor dtype" error rather than silently mis-loading — fail loud, not
    fail silent.

**Phase 6 — End-to-end verification and docs.**
Full quantized run of the real 27B checkpoint; compare output against the Phase-2 bf16 baseline for
coherence/consistency; add to `scripts/run_model_sweep.py`; write up in `docs/inference_report.md`.

*Status: checkpoint downloaded and runs end-to-end without loader/architecture errors; generation
output is not yet coherent. Not done -- see below.*

The real `unsloth/Qwen3.8-27B-NVFP4` checkpoint (22GB `model.safetensors` + 850MB `model_mtp.safetensors`)
was downloaded and pointed at celeg's HF cache layout. Getting it to load at all surfaced six real,
generic bugs -- none of them checkpoint-specific hacks, all confirmed against the real on-disk data,
all with regression coverage added, `ctest` 91/91 green throughout:

1. **`layer_types: "linear_attention"` was an unrecognized token.** `parse_attention_pattern`
   (`src/model/inference/inventory.cpp`) had synonyms for `gdn`/`mamba`/`conv`/etc. but not the literal
   string this checkpoint's `layer_types` actually uses. Added the synonym; added `layer_types` to
   `automatic_inference_test.cpp`'s synthetic qwen3.5 checkpoint (it previously didn't set this key
   at all, so this whole code path had zero coverage for the real per-layer-schedule case).
2. **Tensor-inventory rank cap of 4 rejected the vision patch-embed conv.** `model.visual.patch_embed.proj.weight`
   is legitimately rank 5 (`[hidden, channels, temporal, patch_h, patch_w]`) and
   `SafetensorProjectionProvider` already requires exactly rank 5 -- `build_tensor_inventory`'s blanket
   `shape.size() > 4` was simply too strict. Raised the cap to 5.
3. **`layer_has_feed_forward` / `find_unique` / `infer_intermediate_sizes` only recognized dense
   `<name>.weight` tensors.** NVFP4-packed weights replace that literal tensor with
   `<name>.weight_packed` + sidecars -- the base name never exists on disk. Fixed generically at the
   `TensorInventory` level (`src/model/inference/inventory.cpp`): a "derived" inventory entry is now
   synthesized under the base name with the logical dense shape whenever the NVFP4 sidecar triple is
   present, exactly mirroring the pre-existing INT4/INT8 `_packed` derivation this file already did.
   One inventory-level fix covers every caller that does a plain name lookup.
4. **`repository_has_tensor` (the actual weight-loading name-resolution check, distinct from the
   inventory-level fix above) didn't know about FP8/NVFP4 packed sidecars.** Added
   `has_packed_fp8_matrix`/`has_packed_nvfp4_matrix` alongside the existing `has_packed_int4_matrix`
   check in `src/model/weight_plan.cpp`.
5. **Dense (non-MoE) MLP gate+up fusion (`load_concat_linear_weight`) had no FP8/NVFP4 branch**, only
   dense/GGUF/int8. This checkpoint's ordinary (non-MoE) layers always go through this fused-w13 path,
   so it's not an edge case. Added both: FP8 concat is a straightforward per-row-scale row-stack
   (identical shape to the existing int8 branch). NVFP4 concat requires the two parts to share one
   `global_scale`/`input_global_scale` (verified against the real checkpoint: every layer's gate_proj
   and up_proj have bit-identical global scales -- consistent with being calibrated together) --
   fails loudly rather than silently reconciling if that assumption is ever violated.
6. **Tokenizer BOS resolution and `eos_token_id` metadata were both wrong for this tokenizer.**
   `tokenizer_json_loader.cpp`'s "`<|endoftext|>` doubles as BOS" heuristic was gated on the vocabulary
   having no separate EOS-family token either -- but this tokenizer has both a real chat EOS
   (`<|im_end|>`) *and* a config-declared `bos_token_id` of `<|endoftext|>` at the same time (a
   Qwen-family pattern). Split the heuristic so BOS assignment no longer depends on whether an
   explicit EOS was found. Separately, `config.json`'s `text_config.eos_token_id` is a single id
   (248044) but `generation_config.json`'s `eos_token_id` is the complete `[248046, 248044]` stop set
   actually used at generation time and HF's own tooling treats it as authoritative over `config.json`.
   Taught `catalog.cpp` to merge `generation_config.json`'s `bos_token_id`/`eos_token_id` into metadata
   at the unscoped key (checked before any `text_config.*` fallback) -- a generically useful fix, not
   specific to this checkpoint.

With all six fixed, the checkpoint loads completely: `--memory-report` shows ~20.6GB resident (matches
the expected footprint for a mixed FP8/NVFP4 27B model; was initially misreported as 2.4GB because
`SharedModelWeights::memory_bytes()` summed the pre-existing storage buffers but not the three new
FP8/NVFP4 `DeviceBuffer`s added in Phase 5 -- fixed in `src/backend/cuda/model/weights.cpp`), and
`--print-config` confirms the resolved topology matches the checkpoint exactly:
`layers=64 attention_layers=16 gated_delta_layers=48` (the 3:1 hybrid schedule).

Generation, however, was initially completely degenerate (`结构设计` repeated indefinitely regardless
of prompt). Root-caused to a **real, confirmed sign error in the NVFP4 dequantization formula**,
present since Phase 3/4 and invisible to `cuda_kernels_test.cu`'s existing NVFP4 unit test because
that test only exercises `weight_global_scale=1.5`/`act_global_scale=1.0` -- neither value is large
enough to expose the bug, and the *activation* side is always 1.0 there. celeg's kernels computed
`dequant = e2m1_code * block_scale * global_scale`; the correct compressed-tensors/NVIDIA convention
(`global_scale = FP8_MAX * FP4_MAX / tensor_amax`, i.e. a *larger* global_scale means a *smaller*
tensor) requires `dequant = e2m1_code * block_scale / global_scale`. Verified directly against the
real checkpoint's bytes before changing anything: dequantizing `layers.0.mlp.gate_proj` by hand in
Python, the multiply convention produces weight magnitudes in the hundreds of thousands (obviously
wrong for a transformer weight); the divide convention produces magnitudes around 0.001-0.02 (exactly
the expected range). Fixed in three places that all shared the same inverted convention:
`dequant_nvfp4_kernel` and `quantize_e2m1_per_block_kernel` (`src/backend/cuda/kernels/linear.cuh`),
and the primary cuBLASLt-path post-multiply in `gemm_dispatcher.cpp` (now `1.0f / (weight.global_scale
* weight.input_global_scale)` instead of the product). Updated `cuda_kernels_test.cu`'s scalar
reference to match (it was self-consistently testing the wrong direction, which is why it passed
despite the bug).

After this fix, real-checkpoint generation changed qualitatively -- no longer a fixed repeating token
loop, higher-confidence (larger) logit margins, and the first generated token decodes as valid,
on-topic-adjacent text (Chinese characters for "Canada", not noise) -- but subsequent tokens still
include invalid/replacement-character byte sequences, so generation is not yet coherent. This is
concrete evidence the scale-direction fix was real and directionally correct, but at least one more
issue remains (candidates, not yet isolated: a residual NVFP4/FP8 precision or swizzle issue at scale
across 64 layers, a tokenizer BPE byte-fallback decode edge case independent of quantization, or an
architecture-level issue specific to this checkpoint's exact attention/gated-deltanet configuration
that the Phase 2 synthetic tests didn't exercise). Isolating it further needs either a Python
reference dequantizer extended to a full single-layer numerical comparison, or `--dump-logits` diffing
against a real bf16 baseline run -- neither done yet.

**A separate, real, generic bug found and fixed while investigating the byte-garbage tail**: the CLI
streaming print loops in both `src/app/cuda/main.cpp` and `src/app/cpu/main.cpp` decoded and printed
each new token in isolation (`tokenizer->decode({next}, true)`), with no buffering across token
boundaries. Byte-level BPE routinely splits one multi-byte UTF-8 codepoint (any non-ASCII output, e.g.
CJK) across two or more token ids, so printing token-by-token produces genuinely invalid byte sequences
even when the underlying token ids are correct -- a display bug indistinguishable from a computation
bug by eyeballing terminal output. `src/serve/chat_generation.cpp`'s server-side path already handled
this correctly (buffers and only emits the longest complete-UTF-8 prefix, via a local
`complete_utf8_prefix` helper); extracted that helper to a shared `include/celeg/text/utf8.hpp` and
applied the identical buffering fix to both CLI print loops. Confirmed independent of the checkpoint --
this bug exists for any model producing multi-token UTF-8 output, not just Qwen3.5. Kept as a real fix
regardless of the outcome below.

That fix did **not** resolve incoherence, however -- re-running the real checkpoint after the fix still
produces near-uniform, low-margin logits at the very first prefill-derived token (`--print-top 5` shows
a ~0.3-logit spread across the top 5 candidates, essentially noise-level for a 27B model), and generation
still collapses into a repeating-token loop after a few steps. This rules out the streaming-decode bug
and the sampling loop as the cause of the previously-observed byte garbage, and narrows the remaining
problem to the forward pass itself (something upstream of logits is producing near-random output even
on the very first decode step). Checked and ruled out as an easy lever: `--weight-mode bf16` has no
effect on this checkpoint's per-tensor FP8/NVFP4 storage (the flag doesn't reach `weight_plan.cpp`'s
self-describing quant-format resolution at all, confirming the plan's own noted risk that there is no
dequant-on-load path to a true bf16 baseline for this checkpoint). Re-derived the NVFP4 native
cuBLASLt block-scaled path's `total_scale = 1/(global_w * global_a)` from first principles against the
already-verified weight dequant convention and confirmed it is self-consistent -- not the bug. The
remaining candidates are unchanged from before (residual NVFP4/FP8 precision/swizzle issue, or an
architecture-level bug in the gated-deltanet/attention/MRoPE path specific to this checkpoint's exact
configuration) and isolating further needs per-layer activation tracing against a Python reference,
which has not been done.

Not yet done: `scripts/run_model_sweep.py` entry, `docs/inference_report.md` write-up (both explicitly
deferred until generation is actually coherent -- no point recording a broken baseline).

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
- Phase 5 (done): `tests/packed/{fp8,nvfp4}_test.cpp` — `has_packed_*`/`load_packed_*` against a
  synthetic in-memory repository, including the "sidecar missing → not detected" negative case and
  (NVFP4) both with and without an `input_global_scale` sidecar. Full `ctest` (91/91) green throughout.
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
