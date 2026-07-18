# Validation record — v0.0.20

Completed in the packaging environment:

```text
CPU-only Release configure/build             PASS
CTest                                        PASS — 31/31
parallel paged attention vs sequential       PASS
FP32/BF16 page lifecycle                     PASS
prefix radix exact/partial lookup             PASS
partial-page COW isolation                    PASS
reference counts after cache destruction      PASS
NUMA discovery and best-effort bind path      PASS
CPU C API v1-v5 header compatibility          PASS
ASan/UBSan selected CPU paths                  PASS
CPU shared-library linkage                    PASS
CPU symbols exported                          PASS — 50 total / 19 engine
CUDA-free shared-library dependencies          PASS
C examples and validation tools                PASS
shell and Python syntax                        PASS
```

The reference exporter and C++ comparator were built and syntax-checked. They
were not run against `LiquidAI/LFM2.5-230M` because the official checkpoint was
not available in this environment.

Therefore this release does **not** claim:

- official-logit parity;
- generated-text quality;
- TTFT, ITL or tokens/s for the complete checkpoint;
- effective NUMA locality on the user's host;
- CUDA runtime execution.
