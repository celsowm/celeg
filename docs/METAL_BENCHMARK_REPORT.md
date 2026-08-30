# Metal benchmark baseline

This is the first reproducible physical-M5 baseline for the native Metal
backend. Re-run it with:

```text
python benchmarks/run_metal_bench.py \
  benchmarks/manifests/metal_lfm25_350m_q4_k_m.json
```

## Recorded run

| Field | Value |
| --- | --- |
| Host | Apple M5, macOS 26.5.1, arm64 |
| Backend | Metal native, runtime MSL compilation |
| Checkpoint | `LiquidAI/LFM2.5-350M-GGUF` / `LFM2.5-350M-Q4_K_M.gguf` |
| Context | 128 |
| Prompt tokens | 32 synthetic tokens |
| Direct decode tokens | 8 synthetic tokens |
| Warm-up repetitions | 1 |
| Measured repetitions | 3 |
| Prefill | 39.7687 tokens/s, 804.653 ms |
| Decode | 35.9105 tokens/s, 222.776 ms |

The benchmark uses a predetermined token stream for decode, so sampling policy
and vocabulary reduction are excluded from the timing. The smoke path separately
validates greedy generation and the C API against the same cached Safetensors
and GGUF model families.
