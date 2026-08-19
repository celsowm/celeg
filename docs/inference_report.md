# CELEG Inference Report

**Date**: 2025-08-19  
**Build**: linux-cpu-release, linux-cuda-release  
**Models tested**: 14 models from HF cache

---

## Model Summary

| Model | Format | CUDA Status | CPU Status |
|-------|--------|-------------|------------|
| LiquidAI/LFM2.5-230M | safetensors | ✅ Works | ✅ Works |
| LiquidAI/LFM2.5-230M-GGUF | GGUF (q4_k, q6_k) | ✅ Works | N/A |
| LiquidAI/LFM2.5-350M | safetensors | ✅ Works | ✅ Works |
| LiquidAI/LFM2.5-350M-GGUF | GGUF (q4_k, q6_k) | ✅ Works | N/A |
| LiquidAI/LFM2.5-VL-450M | safetensors | ✅ Works (repetitive output) | ✅ Works (repetitive output) |
| ibm-granite/granite-4.1-3b | safetensors | ✅ Works | ✅ Works |
| Nanbeige/Nanbeige4.2-3B | safetensors | ❌ Garbage output | ✅ Works |
| bartowski/Nanbeige_Nanbeige4.2-3B-GGUF | GGUF | ❌ BPE vocab mismatch | N/A |
| flwrlabs/Lizzy-7B | safetensors | ❌ Empty output | N/A |
| flwrlabs/Lizzy-7B-GGUF | GGUF (q4_k, q6_k) | ✅ Works | N/A |
| google/gemma-4-E4B | safetensors | ❌ Audio tower invalid tensors | N/A |
| inclusionAI/Ling-3.0-tiny | safetensors | ✅ Works | ✅ Works |
| openbmb/MiniCPM5-1B | safetensors | ✅ Works | ✅ Works |
| openbmb/MiniCPM5-1B-GGUF | GGUF (q4_k, q6_k) | ✅ Works | N/A |

---

## CUDA Benchmarks (--benchmark-decode 50, --no-cuda-graph)

| Model | Prefill Tokens | Prefill (ms) | Prefill tok/s | Decode (ms) | Decode ms/tok | Decode tok/s |
|-------|---------------|--------------|---------------|-------------|---------------|--------------|
| LFM2.5-230M | 10 | 16.57 | 603.7 | 55.91 | 1.12 | 894.3 |
| LFM2.5-230M-GGUF | 10 | 13.51 | 740.3 | 52.02 | 1.04 | 961.2 |
| LFM2.5-350M | 10 | 16.72 | 598.1 | 67.76 | 1.36 | 737.9 |
| LFM2.5-350M-GGUF | 10 | 14.37 | 695.7 | 59.77 | 1.20 | 836.6 |
| LFM2.5-VL-450M | 10 | ~15 | ~666 | ~ | ~ | ~ (repetitive) |
| granite-4.1-3b | 9 | 34.53 | 260.7 | 418.06 | 8.36 | 119.6 |
| Lizzy-7B-GGUF | 184 | 61.06 | 3013.5 | 404.22 | 8.08 | 123.7 |
| Ling-3.0-tiny | 32 | 310.38 | 103.1 | 1115.36 | 22.31 | 44.8 |
| MiniCPM5-1B | 14 | 19.05 | 734.9 | 165.50 | 3.31 | 302.1 |
| MiniCPM5-1B-GGUF | 18 | 16.80 | 1071.4 | 139.39 | 2.79 | 358.7 |

---

## CPU Benchmarks (native q4-group-32, avx-vnni)

| Model | Prefill Tokens | Prefill (ms) | Prefill tok/s | Decode (ms) | Decode ms/tok | Decode tok/s |
|-------|---------------|--------------|---------------|-------------|---------------|--------------|
| LFM2.5-230M | 10 | 76.50 | 130.7 | 188.60 | 9.93 | 100.7 |
| LFM2.5-350M | 10 | 104.28 | 95.9 | 196.99 | 12.31 | 81.2 |
| LFM2.5-VL-450M | 10 | 44.82 | 223.1 | 122.62 | 6.13 | 163.1 |
| granite-4.1-3b | 9 | 541.34 | 16.6 | 651.80 | 65.18 | 15.3 |
| Ling-3.0-tiny | 32 | 3330.31 | 9.6 | 2401.38 | 120.07 | 8.3 |
| MiniCPM5-1B | 14 | 323.39 | 43.3 | 573.71 | 28.69 | 34.9 |
| Nanbeige4.2-3B | 37 | 1269.48 | 29.1 | 1348.62 | 67.43 | 14.8 |

---

## CUDA vs CPU Speedup (Decode tok/s)

| Model | CUDA tok/s | CPU tok/s | Speedup |
|-------|-----------|-----------|---------|
| LFM2.5-230M | 894.3 | 100.7 | **8.9x** |
| LFM2.5-350M | 737.9 | 81.2 | **9.1x** |
| LFM2.5-VL-450M | ~ (rep) | 163.1 | N/A |
| granite-4.1-3b | 119.6 | 15.3 | **7.8x** |
| Ling-3.0-tiny | 44.8 | 8.3 | **5.4x** |
| MiniCPM5-1B | 302.1 | 34.9 | **8.7x** |
| Nanbeige4.2-3B | (garbage) | 14.8 | N/A |

---

## Known Issues

1. **Nanbeige/Nanbeige4.2-3B (CUDA)**: Produces garbage output despite correct tokenizer vocab size (166107) matching HF. CPU works correctly. Likely CUDA kernel issue.

2. **bartowski/Nanbeige_Nanbeige4.2-3B-GGUF**: "BPE produced token absent from vocabulary" - GGUF tokenizer vocab doesn't match HF tokenizer vocab.

3. **flwrlabs/Lizzy-7B (CUDA)**: Produces empty output on CUDA. CPU not tested (no safetensors in cache).

4. **google/gemma-4-E4B**: Multimodal model with audio tower - tensor inventory contains invalid tensor metadata (model.audio_tower.layers.0.feed_forward1.ffw_layer_1.input_max).

5. **LiquidAI/LFM2.5-VL-450M**: Works on both backends but produces repetitive output ("RawRawRaw..." on CPU, "inherited inherited..." on CUDA). Vision model, likely needs special handling.

---

## Fixes Applied

1. **Chat template fallback**: Added support for loading `chat_template` from `tokenizer_config.json` (fixes Nanbeige/Nanbeige4.2-3B).

2. **rope_theta aliases**: Extended search to include `rope_parameters.full_attention.rope_theta`, `rope_parameters.sliding_attention.rope_theta`, and `text_config.*` variants (fixes google/gemma-4-E4B rope_theta error).

3. **Vision model feed-forward norm**: Allow layers without FFN to have norms for vision/encoder architectures (fixes LFM2.5-VL-450M).

4. **Tokenizer vocab size propagation**: Added `tokenizer_vocab_size` to tokenizer definition and passed to CUDA model for logits masking.

---

## Notes

- All GGUF models use `weight_mode=auto` (q4_k/q6_k quantization)
- CPU builds use native q4-group-32 quantization with AVX-VNNI
- CUDA benchmarks run with `--no-cuda-graph` for accurate per-token measurement
- CPU backend auto-selects ISA (AVX-VNNI on this machine)
- Pack files cached in `~/.cache/celeg/`