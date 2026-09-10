# CPU/CUDA model sweep

Status: complete for the 15 cached artifacts exercised by
`scripts/run_model_sweep.py` (14 original + `google/gemma-4-E4B-it`,
whose weights were fetched into the local HF cache on 2026-09-10).

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

14 of 15 artifacts answer correctly on both backends (`google/gemma-4-E4B`
is a base checkpoint scored by reference parity, not by answer; its
instruct sibling `google/gemma-4-E4B-it` answers correctly on both).

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
- **CPU: diverged (pre-existing, not a regression).** With
  `--cpu-weight-format bf16` the first generated token is `<b>` (id 200) where
  the reference and CUDA produce `What` (id 3689); prefill logits show cosine
  0.46 vs CUDA with collapsed magnitudes (~14 vs ~30 top logit). The
  divergence is deterministic across thread counts and prefill-chunk sizes,
  silent under `CELEG_STRICT_SEMANTICS=1`, and reproduces with the pre-pull
  Sept-7 CPU binary, so it predates the current HEAD. It is also
  weight-sensitive rather than structural: tensor names (2130) and
  `text_config` are identical between base and `-it`, yet `-it` answers
  correctly on CPU (prefill cosine 0.96 vs CUDA, same argmax). Suspect: CPU
  attention numerics on this architecture (42-layer alternating
  sliding-window-512 / full attention, GQA 8Q/2KV, head_dim 256); needs a
  dedicated investigation, tracked as follow-up work.

The earlier "BF16 restores CPU parity" claim is corrected by the above: BF16
mode fixed the garbled Q4 output but the CPU backend still does not reproduce
the greedy reference on this checkpoint. The sweep enables bf16 for both
gemma rows automatically, and `scripts/run_model_sweep.py` now emits
`PARITY-CORRECT` / `PARITY-DIVERGED` verdicts for base checkpoints.

## Gemma-4 E4B-it (instruct checkpoint)

`google/gemma-4-E4B-it` ships a real `chat_template.jinja`
(`chat.template=checkpoint-metadata`, fingerprint `52b103a5182a6a53`) and
answers `The capital of France is Paris.` on both backends under the default
sweep settings.

## Fixes captured during this sweep

- **Ling-3.0-tiny CUDA runs.** Previously `cuda_init_failed`; now runs and
  answers correctly on CUDA (21 s) and CPU (12 s).
- **Composition-heterogeneous Q/K norms (VL-450M).** Q/K norm tensors under the
  nested `model.language_model.layers.N.self_attn.{q,k}_layernorm.weight` prefix
  now bind.
- **Jinja whitespace parity.** The chat-template renderer now honors
  `trim_blocks` and `lstrip_blocks`, matching
  `tokenizer.apply_chat_template` byte-for-byte for every sweep template.
- **Sweep classifier.** The generated completion is read from the whole stdout,
  not the last line, so thinking-style answers are scored correctly.

## Build and tests

- `python scripts/dev.py verify --backend cpu` — PASS (92/92)
- `python scripts/dev.py verify --backend cuda` — PASS