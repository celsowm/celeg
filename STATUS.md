# Implementation status — v0.0.20

## CPU backend

Implemented:

- Q4 groupwise standalone and packed execution;
- scalar, AVX2, AVX-VNNI and AVX-512 VNNI linear paths;
- layer-major chunked prefill;
- continuous multi-request scheduling and packed decode;
- physical FP32/BF16 KV pages;
- adaptive tile-parallel paged GQA;
- radix longest-prefix cache with complete ShortConv snapshots;
- partial-page copy-on-write;
- LRU cache limits by entry and bytes;
- NUMA node discovery, request placement and best-effort page binding;
- CPU C API v5 and backward-compatible v1-v4 entry points;
- offline official-logit export/comparison tools.

Not yet implemented:

- AMX-INT8, ARM DotProd/I8MM/SME2 executors;
- NUMA weight replication or node-local worker sub-pools;
- INT8 CPU KV pages;
- vectorized attention-specific microkernels;
- physical official-checkpoint parity and end-to-end performance results in the
  packaging environment.

## CUDA backend

CUDA sources and prior features remain in the tree. This release was validated
primarily through the CPU-only build because no physical NVIDIA GPU or complete
CUDA execution environment was available.
