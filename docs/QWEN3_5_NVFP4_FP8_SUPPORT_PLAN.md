# Support unsloth/Qwen3.8-27B-NVFP4 on CUDA: qwen3_5 linear attention + FP8 + NVFP4

Status: **Not started.** Planned 2026-08-20, revised 2026-08-20 after verifying every claim against
the live checkpoint (`config.json`, `model.safetensors.index.json`) and the current tree. This is a
multi-phase, multi-session effort — see Sequencing. Update the phase checklist as work lands.

- [x] Phase 1 — per-tensor quant-format infrastructure (pure refactor)
- [ ] Phase 2 — `linear_attn` binding + vision/MTP fit, running in bf16
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

**Phase 2 — Architecture fit, running in bf16 first (no quantization yet).**
Get the checkpoint loading and generating *correct* text with everything upconverted to bf16
(`--weight-mode bf16`, ignoring `quantization_config` entirely) as the correctness baseline before
any quant kernel work. Concretely:
  - New `linear_attn` name dialect in `rules_recurrent.cpp`, reusing `FusedGatedDeltaRule`'s
    semantics and roles via the mapping table above. Preferred shape: factor the fused rule's
    tensor-name set into a small alias table and add the `model.language_model.layers.N.linear_attn.*`
    entry, rather than copy-pasting a fourth rule class. Confirm shapes from the safetensors header
    first (the index has no shapes); confirm `output_gate_type: "swish"` matches the existing gate
    semantics and that `linear_num_value_heads` (48) ≠ `linear_num_key_heads` (16) is already handled
    by `m.gated_delta.{key,value}_heads` — `metadata.cpp:510` currently aliases both from
    `num_attention_heads`/`num_heads_for_linear_attn`, so the `linear_num_{key,value}_heads` /
    `linear_{key,value}_head_dim` / `linear_conv_kernel_dim` keys likely need adding to those alias
    lists.
  - Full-attention layers: verify `attn_output_gate: true` + `partial_rotary_factor: 0.25` +
    interleaved mRoPE resolve correctly together (each is supported individually — see
    `rules_attention.cpp:265,428` and `cpu_mrope_test.cpp` — the combination is untested).
  - Vision: derive the merger output width from `merger.linear_fc2`'s shape instead of the
    hardcoded 2048 in `safetensor_projection.cpp`; then run `qwen35_vision_test` with
    `CELEG_QWEN35_MODEL` pointed at the real checkpoint (the test currently skips when unset, and its
    `embedding.width == 2048` assertions must become shape-derived too).
  - Verify the existing MTP "one auxiliary attention layer" constraint accepts `mtp.layers.0.*` plus
    `mtp.fc`/`mtp.pre_fc_norm_*`; extend only if it doesn't. Simply not loading MTP is an acceptable
    Phase-2 fallback — it is a speculative-decode accelerator, not required for correct output.
  - Ensure unbound `k_scale`/`v_scale` tensors are tolerated by the loader.
  - Only add an architecture descriptor JSON (`src/model/descriptor/`) if generic inference rules
    genuinely cannot express this config shape; current evidence says they can.

**Phase 3 — FP8 W8A8 kernel.**
New `Fp8LinearStorage` (per-channel fp32 scale + e4m3 packed weight), new
`LinearKernelKind::Fp8W8A8`, a dynamic **per-token** activation-quantization kernel (modeled on
`launch_quantize_q8_1`) producing e4m3 activations, and a cuBLASLt fp8 matmul (`CUDA_R_8F_E4M3`
operands, appropriate compute/scale types). **Spike the cuBLASLt call standalone first** (throwaway
scratchpad program, not committed) — celeg has zero existing fp8 cuBLASLt usage to copy from.
Validate against a small synthetic fp8 GEMM with a scalar reference before wiring into the loader.

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
- Phase 2: a synthetic-checkpoint resolution test for the `linear_attn` dialect (follow
  `cpu_mrope_test.cpp`'s pattern — a tiny hand-written `qwen3_5` checkpoint with one linear-attention
  and one full-attention layer); shape-derived assertions in `qwen35_vision_test`; an MTP
  compatibility check. Non-regression: existing LFM2/MiniCPM/Nanbeige models on both backends
  (`ctest` + relevant sweep entries).
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
- **`linear_attn` shapes are inferred from config, not read from the checkpoint.** The name mapping
  onto the fused gated-DeltaNet dialect is high-confidence, but the exact packing order inside
  `in_proj_qkv` and the conv1d layout must be read from the safetensors header before binding work.
  A packing-order mismatch produces plausible-looking but wrong output — the expensive failure mode.
- **Full-attention feature combination is untested**: output gate + partial rotary (0.25) +
  interleaved mRoPE together.
- **Vision merger width is hardcoded to 2048** and this model needs 5120; the fix is small but the
  test's assertions are hardcoded to the same constant, so both move together.
- **MTP constraint fit is unverified** — confirm before assuming no MTP code changes; dropping MTP is
  an acceptable fallback.
- **cuBLASLt fp8 and NVFP4 block-scaled APIs are both unproven in this codebase** — each phase
  includes a standalone spike specifically to de-risk this before integration.
- **No CUTLASS fallback planned** — if cuBLASLt proves unworkable for either shape, that is a
  materially larger re-scope requiring a check-in with the user, not a silent substitution.
- **Total scope is still large** — realistically multiple sessions; each phase is an independently
  valuable, independently verifiable unit rather than one unreviewable change.
