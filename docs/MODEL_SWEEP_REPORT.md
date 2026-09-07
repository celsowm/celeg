# CPU/CUDA model sweep

Status: complete for the 14 cached artifacts exercised by
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
| `google/gemma-4-E4B` Safetensors | parity (see below) | parity (see below) |
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

13 of 14 artifacts answer correctly on both backends.

## Gemma-4 E4B (base checkpoint)

`google/gemma-4-E4B` is a *base* model, not an instruct checkpoint: it echoes
and rephrases the prompt rather than answering it. Both backends reproduce the
Hugging Face greedy reference byte-for-byte on the opening tokens, so the row
is scored as parity-correct, not a defect.

The CPU backend initially produced garbled output on this checkpoint: its tower
weights drifted under group-32 Q4 quantization. The BF16 weight mode
(`--cpu-weight-format bf16`, `CELEG_CPU_WEIGHT_BF16`) keeps the weights lossless
and restores parity; the sweep enables it for this model automatically.

## Fixes captured during this sweep

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