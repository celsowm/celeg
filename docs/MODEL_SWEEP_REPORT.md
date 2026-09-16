# CPU/CUDA model sweep

Status: complete for the 16 cached artifacts exercised by
`scripts/run_model_sweep.py`.
## Environment

- Platform: Linux
- CPU: Intel Core i9-14900KF, AVX-VNNI (`out/linux-cpu-relwithdebinfo/celeg-cpu-run`)
- GPU: NVIDIA GeForce RTX 5090, 32 GB (`out/linux-cuda-relwithdebinfo/celeg-run`)
- Prompt: `What is the capital of France?`
- Generation: greedy, temperature 0, top-k 1 (300-token budget for thinking-style
  templates, 20 otherwise)
- Cache: local `~/.cache/huggingface/hub`; the remote Hub is never modified

## Result

| Model / artifact | CPU | CUDA |
|---|---|---|
| `LiquidAI/LFM2.5-230M` Safetensors | correct | correct |
| `LiquidAI/LFM2.5-350M` Safetensors | correct | correct |
| `LiquidAI/LFM2.5-VL-450M` Safetensors | correct | correct |
| `google/gemma-4-E4B` Safetensors | diverged (see below) | parity (see below) |
| `google/gemma-4-E4B-it` Safetensors | correct | correct |
| `ibm-granite/granite-4.1-3b` Safetensors | correct | correct |
| `inclusionAI/Ling-3.0-tiny` Safetensors | correct | correct |
| `openbmb/MiniCPM5-1B` Safetensors | correct | correct |
| `Nanbeige/Nanbeige4.2-3B` Safetensors | correct | correct |
| `flwrlabs/Lizzy-7B` Safetensors | correct | correct |
| `LiquidAI/LFM2.5-230M` GGUF | correct | correct |
| `LiquidAI/LFM2.5-350M` GGUF | correct | correct |
| `flwrlabs/Lizzy-7B` GGUF | correct | correct |
| `bartowski/Nanbeige_Nanbeige4.2-3B` GGUF | correct | correct |
| `openbmb/MiniCPM5-1B` GGUF | correct | correct |
| `unsloth/Qwen3.8-27B-NVFP4` Safetensors | correct | garbled (see below) |

15 of 16 cached artifacts answer correctly on both backends
(`google/gemma-4-E4B` is a base checkpoint scored by reference parity,
not by answer; its instruct sibling `google/gemma-4-E4B-it` answers
correctly on both; `unsloth/Qwen3.8-27B-NVFP4` answers correctly on CPU
but generates garbled output on CUDA, see below).

## Gemma-4 E4B (base checkpoint)

`google/gemma-4-E4B` is a *base* model, not an instruct checkpoint: it echoes
and rephrases the prompt rather than answering it. The sweep therefore scores
it by token parity against a recorded Hugging Face greedy reference
(`scripts/gemma_base_parity_reference.json`: BF16 `transformers` generate from
the exact 16-id celeg prompt prefix, 64 new tokens) instead of the `"paris"`
substring check. Two settings matter for the comparison: the run must disable
celeg's default 1.05 repetition penalty (`--repetition-penalty 1.0`, which
otherwise steers the base checkpoint off its greedy echo loop), and the
checkpoint ships no `chat_template.jinja`, so both backends render the prompt
through the turn-delimited template inferred from tokenizer evidence
(`chat.template=tokenizer-inference`, fingerprint `3003c14a01705e0d`).

- **CUDA: parity confirmed.** All 64 generated ids match the reference exactly
  (pure `What is the capital of France?\n` echo loop).
- **CPU: parity confirmed after the partial-rotary fix below**
  (2026-09-16 sweep: 16/16 on both backends). Previously diverged: with
  `--cpu-weight-format bf16` the first generated token was `<b>` (id 200)
  where the reference and CUDA produce `What` (id 3689); prefill logits
  showed cosine 0.46 vs CUDA with collapsed magnitudes (~14 vs ~30 top
  logit). Deterministic across thread counts and chunk sizes, silent under
  `CELEG_STRICT_SEMANTICS=1`, reproducing with the pre-pull Sept-7 CPU
  binary. Weight-sensitive rather than shape-structural: `-it` (identical
  tensor names and `text_config`) answered correctly on CPU (prefill cosine
  0.96), because the residual stream (~90 norm) dominates the corrupted
  mixer contribution and `-it`'s argmax survives the attenuation.

## Root cause: partial-rotary pairing + unnormalized tail (fixed)

Per-layer CPU-vs-CUDA dumps localized the injection to the full-attention
layers (5, 11, 17, 23, 29, 35, 41 -- exactly the `layer_types` schedule),
starting at layer 5 post-residual (cosine 0.97 vs 1.0000 on layers 0-4), and
a temporary Q/K/V probe plus an HF (`transformers` 5.15) recomputation
pinned it to QK preparation: CPU layer-5 Q had norm 2739 vs 65.5 reference
while V matched at 0.9999. Two defects in the fused QK-norm+RoPE path,
both from treating `rotary_dim` as the whole head instead of a prefix:

1. **Wrong rotation partners.** Gemma-4 full layers use proportional RoPE
   (`theta` 1M, `partial_rotary_factor` 0.25, head_dim 512). HF pads the
   frequency table with zeros to the full head width, so pair `i` mixes
   `(i, i + head_dim / 2)`; celeg paired `(i, i + rotary_dim / 2)`.
2. **Unnormalized tail.** The fused kernel normalized only the rotated
   prefix and left dims past `rotary_dim` as raw GEMM output (the 42x
   blowup); HF normalizes the whole head first.

The fix is per-scaling-convention, applied agnostically on all three
backends: `ProportionalRopeScaling` uses full-dimension pairs, every other
scaling keeps legacy prefix pairs (Ling-3.0-tiny pins
`partial_rotary_factor = 1.0` over its rope prefix and rotates it with
`rotate_half` -- verified in its modeling code -- so a global switch would
have broken it). Concretely: `src/backend/cpu/kernels/rope.cpp`
(norm-whole-head-first, convention-dependent pairing in
`apply_qk_norm_rope_scalar` and `cpu_rope`),
`src/backend/cuda/kernels/rope.cuh` (`dynamic_qk_norm_rope_kernel`
pairing + full-dim-aware remainder norm; the pre-norm already covers the
whole head), and `qk_position.metal` (pairing, mode-6-scoped). M-RoPE and
adjacent-pair kernels are untouched (no sweep model exercises partial
mrope; Qwen3.8's mrope path is unaffected -- confirmed by the unchanged
sweep row). A fixed-pipeline simulation matches HF mixer output at cosine
0.999992, and post-fix dumps agree at >= 0.99965 on all 42 layers.

Regression coverage: `test_proportional_partial_rope_uses_full_dim_pairs`
and `test_default_partial_rope_keeps_prefix_pairs_and_norms_tail` in
`tests/cpu/attention_semantics_test.cpp` (fused + rope-only vs an HF-style
reference; both fail on the old code), plus matching proportional/legacy
blocks in `tests/cuda/attention_tests.cu` (the proportional block fails on
the old kernel with an O(1) mismatch).

## Gemma-4 E4B-it (instruct checkpoint)

`google/gemma-4-E4B-it` ships a real `chat_template.jinja`
(`chat.template=checkpoint-metadata`, fingerprint `52b103a5182a6a53`) and
answers `The capital of France is Paris.` on both backends under the default
sweep settings.

## Qwen3.8-27B-NVFP4 (compressed-tensors FP8 + NVFP4)

`unsloth/Qwen3.8-27B-NVFP4` is a mixed-precision checkpoint: 401 E4M3 weights
(`float-quantized`, per-row scales; all attention/linear-attn projections,
lm_head, layers-56-63 MLPs) plus 168 NVFP4-packed MLPs (`nvfp4-pack-quantized`,
block-16 e4m3 scales, per-tensor F32 global scales), ~12.4B logical params
across a 64-layer 3:1 gated-DeltaNet/full-attention hybrid.

- **CPU: correct** (added 2026-09-13). `CpuWeightCodec` previously rejected
  every F8/U8 tensor (`lm_head.weight` was the first); it now detects the
  packed sidecars via the checkpoint-neutral `has_packed_fp8_matrix` /
  `has_packed_nvfp4_matrix` probes and dequantizes through host float into
  the ordinary dense path (Q4-group pack by default, lossless in bf16 mode).
  Two memory bugs in that same load are also fixed: source pages are now
  released after each packed tensor is consumed (mmap page cache no longer
  holds all 22 GB resident next to the packed weights; 29.8 GB peak RSS
  OOM becomes ~15.8 GB), and the legacy `compressed_checkpoint` guess (any
  name ending in `_packed`) no longer retains a second copy of every matrix
  in a shadow cache when NVFP4 sidecars are present.
  The default run now answers `...The capital of France is **Paris**.` —
  same for `--cpu-q4-group 64`.
- **CUDA: correct** (fixed 2026-09-15; was garbled, rediagnosed below).
  Single-tensor FP8/NVFP4 weights bind the native `Fp8W8A8` / `Nvfp4W4A4`
  cuBLASLt paths end to end, and the default run answers `Paris` (greedy,
  sampled, graph and no-graph decode alike), matching the CPU trajectory.
  The 2026-09-13 "genuine 4-bit precision limits" diagnosis was wrong: the
  garble was a missing packed-gate extraction on the CUDA decode path, not
  quantization noise (see rediagnosis).

### Rediagnosis of the CUDA gap (2026-09-15)

Bisecting every quantized linear to bf16 (per-layer, per-format, and all 16
full-attention layers) never moved the divergence, which exonerated all
quantized linears. CPU-vs-CUDA decode dumps then showed layers 0-2
(gated-DeltaNet) matching at cosine 0.996+ while every full-attention layer
(3, 7, 11, ...) diverged, with layer-3 mixer output at norm ~128 vs ~9 on
CPU. Root cause: this checkpoint packs the attention output gate into
`q_proj` (`[12288, 5120]`, per-head interleaved `[q_h, g_h]`), and
`enqueue_decode_standard_attention` never de-interleaved it -- every query
head ran RoPE/attention over gate-contaminated data and the gate itself was
read from the wrong half (`apply_cuda_graph_attention_gate` assumed
contiguous `[Q|G]` halves). Prefill was unaffected (it extracts correctly),
which is why prefill-only comparisons looked clean. The fix extracts via
`prepare_cuda_token_attention_gate` right after projection and applies via
`apply_cuda_token_attention_gate`; the wrong-convention graph helper is
deleted. Layer-3 mixer output is now norm ~22.5 vs ~24.6 on CPU.
Remaining CUDA-vs-CPU drift is smooth quantization noise (0.97 at layer 0
to ~0.3 at layer 63), tolerated by the model exactly like the CPU's own
Q4-32 vs Q4-64 divergence -- greedy and sampled runs answer `Paris`.
Regression coverage, two levels. `run_packed_gate_decode_tests` (in
`cuda_kernels_test`) drives the exact decode launcher sequence -- per-head
extraction, QK-norm / RoPE prepare, strict + online contiguous decode,
sigmoid gate apply -- at a tiny geometry and at Qwen3.5 scale (24 heads,
head_dim 256, partial rotary 0.25), checks every stage against a host
reference, and requires the halves-split misreading to diverge.
`cuda_packed_gate_model_test` goes one level up: a one-layer synthetic
model with a packed gate (2 heads, so interleave and halves differ)
runs prefill plus decode on CUDA and on the lossless-CPU backend from the
same committed fixture, and asserts logits (max-abs < 0.05, cosine >=
0.999) plus the decoded token. Reverting the
`enqueue_decode_standard_attention` fix makes it fail, restoring makes it
pass -- the caller wiring is covered, not just the kernels. Two lessons
from building it: the CPU reference must run `CpuWeightFormat::Bf16`
(the default Q4 packing drowns the comparison), and fixture values must
push the gate across sigmoid's slope with spread attention mass, else a
broken mixer hides behind sigmoid(0) ~= 0.5 and uniform softmax. The same
session also fixed the `warmup_decode_gemms` K/V offsets, which assumed
the unpacked query width (harmless: warmup outputs are discarded).

### Diagnosis of the CUDA gap (2026-09-13, superseded)

Backing measured evidence before any kernel change, per the AGENTS.md
performance-investigation rule (here applied to correctness):

- **Kernels are fine where tensors are small.** BF16 miniature checkpoints
  (random weights) match CPU at cosine 0.99997–1.00000 across gated-delta
  and full-attention layers, prompt lengths 2 and 29, single layer and
  64-layer stacks; logits agree at cosine 0.9998 at width 5120. A
  miniature with the real layer-0 weights (FP8 tensors dequantized off
  disk, applied as dense bf16) matches at cosine 0.999989.
- **CUDA vs Python reference on real layer 0** (independent torch port of
  the gated-delta kernel, validated at cosine 1.0000000 against CPU on
  miniature data): CUDA bf16-dequantized layer-0 mixer hits cosine
  0.9999997; the Q4-repacked CPU path only reaches 0.9994. Exact kernels
  beat the packed CPU path on identical inputs.
- **Residual gap is quantization, not a kernel bug.** CPU-side A/B:
  `--cpu-q4-group 64` vs default `--cpu-q4-group 32` on otherwise-identical
  weights already diverges to cosine ~0.98 by layer 31 and ~0.72 by layer 63
  — pure weight-rounding precision on both sides. CUDA's per-16/ NVFP4
  confined-precision and the CPU Q4 path diverge by the same mechanism and
  by roughly the same amount; backward accumulation through 64
  attention-free recurrent layers (state ~ `1/decay` amplification) turns
  it into a different final token.
- **Supporting evidence:** bisect scan flags (`CELEG_BF16_LAYERS`,
  `CELEG_BF16_FORMATS`, intersection semantics; see below) show every
  eight-layer fp8 or nvfp4 window stays within cosine ≥ 0.99 of CPU through
  layer 23 and ≥ 0.97 through layer 31 — no cliff step that would mark a
  broken kernel; cross-format substitution (all layers dequantized then
  re-quantized as int8/int4 W8A16/W4A16) also garbles, ruling out a format
  specific gap; `CUBLAS_WORKSPACE_CONFIG` and `CUDA_LAUNCH_BLOCKING=1` do
  not change the outcome (the kernel-to-kernel run-to-run noise at 1e-5
  vs 1e-4 to CPU is a latency artifact, not the bug).

The forward is therefore correct within genuine 4-bit accuracy; this
checkpoint just does not stay coherent under per-16-block FP4 weights on
this implementation — comparable to the Qwen3.5 4-bit situation documented
for Metal (`docs/METAL_ALL_MODELS_CLOSURE_PLAN.md`), and to the int4
limitation recorded for Agnes below. A quality fix is a precision problem
(structured/high-dimensional 4-bit scales or a 6-bit path), not a kernel
bug-hunt. Performance attributes: measured 67 s CPU, 38 s CUDA for the
sweep budget (300-token; the CUDA run is fast once loaded).

### Debug/bisect flags added during this investigation

The CUDA loader now honors two read-only environment flags (no production
behavior change when unset):
- `CELEG_BF16_LAYERS=0-7,56-63` — dequantize the listed layers' packed FP8 /
  NVFP4 weights to plain bf16 at load time (fits the "one-layer-window" and
  "one-format-window" bisect patterns without a special build). `head`
  names the untied `lm_head` projection; `all` covers every layered tensor.
- `CELEG_BF16_FORMATS=fp8,nvfp4` — same for all tensors of a given packed
  format. With both flags set the two conditions intersect (one format
  within a layer window), so a 24-layer NVFP4-bf16 run stays under 30 GB
  and fits the RTX 5090.
- `CELEG_DEBUG_LAYER_STATS` and `CELEG_DEBUG_HIDDEN_DIR` now also dump the
  decode-path hidden state (`cuda_dec_layerN_{mixer-out,post-mlp}.f32`),
  mirroring the prefill `cuda_layerN_*` dumps, for CPU-vs-CUDA decode
  divergence mapping (only with `--no-cuda-graph`, since a synchronous
  copy-out cannot run inside a graph).

## Fixes captured during this sweep

- **Sweep classifier.** The generated completion is read from the whole stdout,
  not the last line, so thinking-style answers are scored correctly.
- **Qwen3.8-27B-NVFP4 loads on CPU** (packed FP8/NVFP4 matrix branches in
  `CpuWeightCodec`, mmap source-page release after consumption, and the
  compressed-checkpoint dedup-cache no longer double-holds NVFP4 matrices).
  See the dedicated section above.

## Build and tests

- `python scripts/dev.py verify --backend cpu` — 106/106
- `python scripts/dev.py verify --backend cuda` — 133/133
- `python scripts/run_model_sweep.py` (2026-09-16, all 16 local checkpoints,
  both backends) — 16/16 correct-or-parity on both backends, including
  `google/gemma-4-E4B` CPU now PARITY-CORRECT after the partial-rotary fix
  and `unsloth/Qwen3.8-27B-NVFP4` OK-CORRECT from the packed-gate fix