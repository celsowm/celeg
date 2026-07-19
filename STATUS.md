# Implementation status — v0.0.21

## Supported variants

| Variant id              | Repo                              | Status      |
|-------------------------|-----------------------------------|-------------|
| `lfm2.5-230m`           | `LiquidAI/LFM2.5-230M`            | Loaded from `config.json`; runtime-selected. |
| `lfm2.5-1.2b-instruct`  | `LiquidAI/LFM2.5-1.2B-Instruct`   | Loaded from `config.json`; runtime-selected. |

Both variants share the same kernel paths; no per-variant `#ifdef` exists in
the kernel sources. Variants are registered in `ModelVariantRegistry` and
selected by matching the parsed `ModelShape`.

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
- CPU C API v6 (legacy v1-v4 entry points removed);
- offline official-logit export/comparison tools.

Not yet implemented:

- AMX-INT8, ARM DotProd/I8MM/SME2 executors;
- NUMA weight replication or node-local worker sub-pools;
- INT8 CPU KV pages;
- vectorized attention-specific microkernels;
- physical official-checkpoint parity and end-to-end performance results in the
  packaging environment.

## CUDA backend

CUDA sources and prior features remain in the tree. Both variants are wired
through `ModelShape` and `ModelVariantRegistry`; the kernel signatures were
already runtime-parameterized and now receive the variant's dimensions at
every call site. This release was validated primarily through the CPU-only
build because no physical NVIDIA GPU or complete CUDA execution environment
was available.

## SOLID posture

- **SRP** — `ModelShape` (immutable topology), `IModelVariant` (identity and
  matching), `IChatTemplate` (formatting), `ExecutionPlan` (kernel selection)
  and the session/diagnostic/persistence facade views are each single-purpose.
- **OCP** — new variants are added by registering a new `IModelVariant`
  subclass; no kernel file edits are required.
- **LSP** — `LfmModel` is the only public CUDA type; 230M and 1.2B-Instruct
  are substitutable through the same `LfmInferenceSession` /
  `LfmDiagnostics` / `LfmPersistence` views.
- **ISP** — `IChatTemplate` exposes only `format(...)`; tokenizer clients do
  not depend on variant metadata.
- **DIP** — kernels depend on the `ModelShape` value passed in, not on
  compile-time topology constants. The former `LfmConfig` constexpr struct has
  been removed.
