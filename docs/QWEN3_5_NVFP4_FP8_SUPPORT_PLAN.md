# Support unsloth/Qwen3.8-27B-NVFP4 on CUDA: new architecture + FP8 + NVFP4

Status: **Not started.** Planned 2026-08-20. This is a multi-phase, multi-session effort — see
Sequencing below. Update the phase checklist as work lands.

- [ ] Phase 1 — per-tensor quant-format infrastructure (pure refactor)
- [ ] Phase 2 — Qwen3.5 architecture support, running in bf16
- [ ] Phase 3 — FP8 W8A8 kernel
- [ ] Phase 4 — NVFP4 W4A4 kernel
- [ ] Phase 5 — wire real `quantization_config` into the per-tensor resolver
- [ ] Phase 6 — end-to-end verification against the real checkpoint + docs

## Context

The user wants `unsloth/Qwen3.8-27B-NVFP4` running on CUDA with real 4-bit execution (not a
bf16 upconvert — a 27B model at bf16 needs ~54GB, more than the 32GB RTX 5090 available here).
Pulling the real `config.json` and `model.safetensors.index.json` revealed this is a much larger
task than "add an NVFP4 kernel":

- **New architecture.** `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`) mixes, per
  layer, a standard `self_attn` and a separate `linear_attn` module (`in_proj_qkv`, `in_proj_z`,
  `out_proj` — only 3 tensors, no separate alpha/beta/dt_bias/A_log/conv), plus a vision tower
  (`model.visual.*`) and a single-layer MTP/speculative-draft module (`mtp.layers.0.*`).
- **Two coexisting quant formats, not one.** MLP layers 0–55 use NVFP4 (`weight_packed` +
  per-16-block fp8 `weight_scale` + per-tensor fp32 `weight_global_scale` + a static
  `input_global_scale` for dynamic-local W4A4 activation quant — this is genuine W4A4, both
  operands quantized). Attention proj, `lm_head`, and MLP layers 56–63 use a **different**
  format: per-channel static FP8 (`e4m3`) weights (`weight` + `weight_scale`) with **dynamic
  per-token FP8 activation quantization** (W8A8). No mixture of the two is optional — both must
  work for this checkpoint to run correctly.
- Checkpoint is 23.4GB total, no MoE experts, fits the 32GB card comfortably once quantized.

Investigation (2026-08-20 session) found the news is better than it first looked:

- **celeg already supports per-layer mixed mixers** (`src/model/descriptor/architecture.cpp`
  reads HF `layer_types` into a `LayerSpec.mixer` variant per layer) — regular attention and
  gated-deltanet-style layers *already* coexist in one model today (this is how Qwen3-Next-style
  hybrids work, not just LFM2). Qwen3.5's `linear_attn` needs a new tensor-binding branch in
  `src/model/inference/rules_recurrent.cpp` (its 3-tensor layout doesn't match the existing
  6-tensor gated-deltanet binding), but not a new mixer-dispatch mechanism.
- **Architecture registration is generic, not a big per-model switch.** New architectures load
  from JSON descriptors (`src/model/descriptor/registration.cpp`) plus generic HF-config-driven
  inference rules — LFM2 itself needed almost no LFM2-specific code.
- **A real, reusable ViT already exists** (`src/runtime/vision/patch_projection.cpp`, used by
  `LiquidAI/LFM2.5-VL-450M` today) — Qwen3.5's vision blocks need new tensor-name bindings, not
  a new vision implementation.
- **MTP already exists** (`src/backend/cuda/model/mtp_execution.cu`), CUDA-only, constrained to
  "exactly one auxiliary full-attention decoder layer" — Qwen3.5 has exactly one MTP layer with
  standard `self_attn`, which plausibly fits this constraint as-is (needs verification, not a
  known blocker).
- **Weight storage is already per-tensor** (`LinearWeight::storage` is a variant of
  `Bf16LinearStorage`/`Int8LinearStorage`/`Int4LinearStorage`/`GgufLinearStorage`, mixed today
  for GGUF-native models). The blocker to mixing FP8 and NVFP4 in one model is that **kernel
  selection is currently global** (`GemmDispatcher` reads `plan.linear_kernel()`, one value for
  the whole model) and the **loader's `weight_mode_` is a single global field**. Both are small,
  well-isolated changes: add a `LinearKernelKind` to `LinearWeight` itself (loader-set,
  per-tensor) and read it in the dispatcher instead of the plan; replace the loader's global
  `weight_mode_` with a per-tensor-name/role format resolver fed by the checkpoint's
  `quantization_config.config_groups[*].targets` regex lists.
- **No existing FP8 support at all** (`CUDA_R_8F_E4M3`/e4m3 — zero hits in `src/backend/cuda/`).
  **One existing dynamic-activation-quant precedent**: `launch_quantize_q8_1`
  (`src/backend/cuda/kernels/mmq.cu`) dynamically quantizes bf16 activations to int8 before the
  GGUF MMQ kernel — the pattern to imitate for both FP8 and NVFP4 activation quantization, but
  the fp8/fp4 kernels themselves are net-new.

## Standing constraint: stay architecture-agnostic

celeg's established rule (see `docs/inference_report.md`: "implemented in the
automatic/architecture-agnostic resolution path ... not as per-model configuration... an
earlier draft ... used [a per-model descriptor] and was deliberately replaced with this generic
version instead, since a JSON file per model family doesn't scale") applies to every phase here:

- Phase 2's `linear_attn`/vision tensor bindings must be driven generically from tensor-name
  patterns and HF `config.json` fields (mirroring how `rules_recurrent.cpp` and
  `layer_bindings.cpp` already key off generic role/name patterns, e.g. the existing `"vision"`/
  `"vl"` substring checks), not a `qwen3_5`-specific special case in the resolution code. Only
  add a new architecture descriptor JSON (`src/model/descriptor/`) if the generic inference
  rules genuinely cannot express Qwen3.5's shape — and even then, keep the descriptor's content
  declarative (shapes/rope config), not a hardcoded behavioral branch.
- Phase 5's `quantization_config` parser must interpret the `config_groups[*].format` +
  `targets` regex mechanism generically (any checkpoint using this compressed-tensors-style
  schema should resolve correctly), not special-cased to this one repo's exact group names or
  layer ranges (e.g. don't hardcode "layers 56-63" — derive quantized-vs-not per tensor from the
  `targets` regex match against the tensor's own name, so a differently-shaped checkpoint using
  the same schema works without new code).
- If a generic rule and a per-model shortcut both solve a step, prefer the generic rule, even if
  it takes more work — this has already paid off once in this codebase (the Nanbeige fix cited
  above) and is treated as a hard constraint, not a style preference.

## Sequencing

This is a multi-phase effort; each phase should be built and verified independently before the
next, since a failure in one is otherwise very hard to isolate from the others. Recommended
order (dependencies flow downward):

**Phase 1 — Per-tensor quant-format infrastructure (foundation for 3 & 4).**
Add `LinearKernelKind` as a field on `LinearWeight` (set by the loader per-tensor, mirroring how
it already knows which storage variant it populated) instead of solely on `CudaExecutionPlan`.
Change `GemmDispatcher::compile_linear_binding` (`src/backend/cuda/runtime/gemm_dispatcher.cpp`)
to read `binding.kernel` from the weight, falling back to the plan's kernel for tensors that
don't set one (keeps every existing single-format model working unchanged). Replace
`WeightLoader`'s single `weight_mode_` (`linear_loader.cpp`) with a per-tensor-name resolver
function — for now this can just be the existing global mode wrapped as a resolver that always
returns the same value, so this phase is a pure refactor with no behavior change, verified by
the full existing `ctest` suite staying green (89/89) with zero new formats yet.

**Phase 2 — Qwen3.5 architecture, running in bf16 first (no quantization yet).**
Get the checkpoint loading and generating *correct* text with everything upconverted to bf16
(`--weight-mode bf16`, ignoring `quantization_config` entirely for this phase) as the
correctness baseline before any quant kernel work — this isolates "does the architecture work"
from "does the quantization work," which matters given three genuinely new pieces are being
added at once. Concretely:
  - New `linear_attn` tensor-binding branch in `src/model/inference/rules_recurrent.cpp` for the
    3-tensor (`in_proj_qkv`/`in_proj_z`/`out_proj`) layout — inspect the real tensor shapes from
    the checkpoint (already have `config.json`/index locally in `/tmp`) to determine the exact
    QKV/gate split and whether a conv/state-decay parameterization is present elsewhere (e.g.
    fused into `in_proj_qkv` or genuinely absent — a linear-attention variant without decay is
    plausible and would simplify this).
  - New vision tensor-name bindings for `model.visual.blocks.N.{attn.qkv,attn.proj,
    mlp.linear_fc1,mlp.linear_fc2}` mapped onto `patch_projection.cpp`'s expected roles.
  - Verify the existing MTP "one auxiliary attention layer" constraint accepts `mtp.layers.0.*`
    unmodified; extend only if it doesn't.
  - A new architecture descriptor JSON (`src/model/descriptor/` — follow the existing descriptor
    pattern) if generic inference rules don't already cover `qwen3_5`'s specific config shape.

**Phase 3 — FP8 W8A8 kernel.**
New `Fp8LinearStorage` (per-channel fp32 or fp8 scale + e4m3 packed weight), new
`LinearKernelKind::Fp8W8A8`, a dynamic per-token activation-quantization kernel (modeled on
`launch_quantize_q8_1`) producing e4m3 activations, and a cuBLASLt fp8 matmul call
(`CUDA_R_8F_E4M3` operands, appropriate compute/scale types — spike this standalone first,
same reasoning as the NVFP4 spike below, since celeg has zero existing fp8 cuBLASLt usage to
copy from). Validate against a small synthetic fp8 GEMM with a scalar reference before wiring
into the loader.

**Phase 4 — NVFP4 W4A4 kernel.**
`Nvfp4LinearStorage` (packed e2m1 + per-16-block `UE4M3` scale + per-tensor fp32 global scale),
`LinearKernelKind::Nvfp4W4A4`, a dynamic per-16-block activation-quantization kernel to NVFP4
(using the static `input_global_scale` from the checkpoint as calibration), and a cuBLASLt
block-scaled matmul call using `CUDA_R_4F_E2M1` operands with
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` scale mode (confirmed available: CUDA 13.2,
RTX 5090/sm_120 both support this). **Spike this cuBLASLt call standalone first** (throwaway
scratchpad program, not committed) to pin down exact descriptor attributes, K-dimension
alignment/padding requirements, and confirm the block-scaled API behaves as documented for this
shape — resolve this empirically, don't guess. Validate against a scalar e2m1-LUT reference.

**Phase 5 — Wire the real `quantization_config` into the Phase-1 per-tensor resolver.**
Parse `config.json`'s `quantization_config.config_groups` (format `mixed-precision`, groups
keyed by `format: "float-quantized"` → FP8 and `format: "nvfp4-pack-quantized"` → NVFP4, each
with a `targets` list of regexes matching tensor names, e.g.
`re:.*self_attn\.(q|k|v|o)_proj$`) into the per-tensor-name resolver from Phase 1, plus the
`ignore` list (vision tower — stays unquantized/bf16). This is what makes the loader actually
pick FP8 vs NVFP4 vs bf16 per tensor for this specific checkpoint, rather than a single global
`--weight-mode`.

**Phase 6 — End-to-end verification and docs.**
Full quantized run of the real 27B checkpoint; compare output against the Phase-2 bf16 baseline
for coherence/consistency; add to `scripts/run_model_sweep.py`; write up in
`docs/inference_report.md`.

## Tests (per phase, added incrementally — not all upfront)
- Phase 1: existing `ctest` suite must stay green with zero behavior change (pure refactor).
- Phase 2: a loader/binding test for the new `linear_attn` tensor layout; a vision tensor-binding
  test; an MTP compatibility check. Non-regression: existing LFM2/MiniCPM/Nanbeige models on both
  backends must stay unaffected (rerun `ctest` + relevant sweep entries).
- Phase 3 & 4: synthetic-weight kernel correctness tests (dequant/matmul vs. scalar reference),
  per format, before wiring into the loader.
- Phase 5: a `quantization_config` parser test against a synthesized minimal `config.json`
  fragment matching the real schema (already captured from the live checkpoint).
- Phase 6: full model run against the real checkpoint, `classify_output()` coherence/correctness
  check (reuse from `scripts/run_model_sweep.py`), VRAM usage sanity check (~14-17GB expected).

## Verification (cumulative, end state)
1. `ninja -j32 && ctest -j8` green throughout (89/89 baseline, growing with new tests).
2. `unsloth/Qwen3.8-27B-NVFP4` downloaded via celeg's existing `--repo` HF fetch (878GB free
   disk confirmed, 23.4GB download — not a concern) and produces coherent, correct output via
   `celeg-run --repo unsloth/Qwen3.8-27B-NVFP4 --prompt "What is the capital of France?"`.
3. VRAM usage in the expected ~14-17GB range, confirming real 4-bit/8-bit storage (not a bf16
   fallback) is actually in effect.
4. Non-regression across the existing model sweep on both backends.

## Risks
- **Three new subsystems at once** (architecture, FP8, NVFP4) is why phases are sequenced with
  Phase 2 deliberately using bf16-only weights first — isolates architecture correctness from
  quantization correctness. Do not skip the bf16 baseline step.
- **`linear_attn`'s exact tensor semantics are inferred, not confirmed** — the real per-tensor
  shapes/roles inside `in_proj_qkv`/`in_proj_z` need inspection before Phase 2 binding work
  starts (config.json/index alone don't give per-tensor shapes; need the safetensors header).
- **MTP constraint fit is unverified** — Phase 2 must confirm `mtp.layers.0.*` satisfies the
  existing "one auxiliary attention layer" assumption before assuming no MTP code changes needed.
- **cuBLASLt fp8 and NVFP4 block-scaled APIs are both unproven in this codebase** — both phases
  include a standalone spike specifically to de-risk this before larger integration.
- **No CUTLASS fallback planned** for either format — if cuBLASLt's APIs prove unworkable for
  either shape, that's a materially larger re-scope (CUTLASS integration) requiring a check-in
  with the user rather than silent substitution.
- **Total scope is large** — realistically multiple sessions; each phase is designed to be a
  independently valuable, independently verifiable unit rather than one big unreviewable change.
