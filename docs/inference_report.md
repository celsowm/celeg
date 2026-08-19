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
| LiquidAI/LFM2.5-VL-450M | safetensors | ❌ Feed-forward norm error | N/A |
| ibm-granite/granite-4.1-3b | safetensors | ✅ Works | ✅ Works |
| Nanbeige/Nanbeige4.2-3B | safetensors | ❌ No chat template | N/A |
| bartowski/Nanbeige_Nanbeige4.2-3B-GGUF | GGUF | ❌ BPE vocab error | N/A |
| flwrlabs/Lizzy-7B | safetensors | ✅ Works (empty output) | N/A |
| flwrlabs/Lizzy-7B-GGUF | GGUF (q4_k, q6_k) | ✅ Works | N/A |
| google/gemma-4-E4B | safetensors | ❌ Missing rope_theta | N/A |
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
| granite-4.1-3b | 9 | 541.34 | 16.6 | 651.80 | 65.18 | 15.3 |
| Ling-3.0-tiny | 32 | 3330.31 | 9.6 | 2401.38 | 120.07 | 8.3 |
| MiniCPM5-1B | 14 | 323.39 | 43.3 | 573.71 | 28.69 | 34.9 |

---

## CUDA vs CPU Speedup (Decode tok/s)

| Model | CUDA tok/s | CPU tok/s | Speedup |
|-------|-----------|-----------|---------|
| LFM2.5-230M | 894.3 | 100.7 | **8.9x** |
| LFM2.5-350M | 737.9 | 81.2 | **9.1x** |
| granite-4.1-3b | 119.6 | 15.3 | **7.8x** |
| Ling-3.0-tiny | 44.8 | 8.3 | **5.4x** |
| MiniCPM5-1B | 302.1 | 34.9 | **8.7x** |

---

## Known Issues

1. **LFM2.5-VL-450M**: "checkpoint exposes feed-forward normalization for a layer with no feed-forward semantics: 0" - Vision model not supported
2. **Nanbeige/Nanbeige4.2-3B**: Missing chat template metadata
3. **bartowski/Nanbeige_Nanbeige4.2-3B-GGUF**: "BPE produced token absent from vocabulary" - tokenizer mismatch
4. **google/gemma-4-E4B**: "checkpoint applies RoPE but does not specify rope_theta"
5. **flwrlabs/Lizzy-7B**: Produces empty output on CUDA

---

## Notes

- All GGUF models use `weight_mode=auto` (q4_k/q6_k quantization)
- CPU builds use native q4-group-32 quantization with AVX-VNNI
- CUDA benchmarks run with `--no-cuda-graph` for accurate per-token measurement
- CPU backend auto-selects ISA (AVX-VNNI on this machine)
- Pack files cached in `~/.cache/celeg/`