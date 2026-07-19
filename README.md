# lfm25-native-cpp

Experimental C++20 inference runtime for LiquidAI LFM2.5 checkpoints, with
independent NVIDIA CUDA and native CPU backends. The runtime reads the
original `safetensors`, `config.json` and `tokenizer.json` files. Python or
another model-serving runtime is not required.

## Supported variants

| Variant id              | HuggingFace repo                       | Hidden | Layers | Q heads | KV heads | Head dim | Vocab  |
|-------------------------|----------------------------------------|-------:|-------:|--------:|---------:|---------:|-------:|
| `lfm2.5-230m`           | `LiquidAI/LFM2.5-230M`                 |   1024 |     14 |      16 |        8 |       64 |  65536 |
| `lfm2.5-1.2b-instruct`  | `LiquidAI/LFM2.5-1.2B-Instruct`        |   2048 |     16 |      32 |        8 |       64 |  65536 |

Variants are selected at runtime from the checkpoint's `config.json` through
`lfm::ModelVariantRegistry`. Adding a new variant does not require editing the
kernels: register a new `IModelVariant` subclass and the runtime will pick it
up automatically (Open/Closed Principle).

## v0.0.21: multi-variant support

The runtime is no longer specialized for the 230M checkpoint. The major
refactor is the introduction of `lfm::ModelShape`, a runtime topology
descriptor that replaces the former `LfmConfig` constexpr struct. Every buffer
size and kernel call now reads dimensions from `ModelShape` rather than from
compile-time constants, so the same binary can execute either the 230M or the
1.2B-Instruct checkpoint.

Additional changes:

- `IModelVariant` / `ModelVariantRegistry` for variant discovery and selection;
- `IChatTemplate` interface with the LFM2 Instruct template selected per
  variant by the tokenizer;
- `PhysicalPagedKvCache` no longer hard-codes 6 attention layers; it accepts
  the variant's attention-layer count and slot table at construction;
- session file header bumped to v2 (`LFMSESS2`) with a variant id field; old
  v1 session files are rejected cleanly;
- CPU C API v6 — the legacy v1-v4 entry points were removed (no retrocompat
  surface is maintained in this release);
- CUDA C API gains `lfm25_model_vocab_size`;
- CMake `LFM_VARIANTS` option lists which variants the build advertises
  (default: `230m;1.2b-instruct`).

The CUDA backend continues to provide quantized weights, paged KV, prefix
reuse, ragged packed prefill, packed decode, continuous scheduling, CUDA
Graphs and cuBLAS/cuBLASLt. When `nvcc` is unavailable, CMake builds only the
CPU targets.

## CPU-only build

```bash
cmake -S . -B build-cpu \
  -DLFM_ENABLE_CUDA=OFF \
  -DLFM_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

Standalone execution (230M):

```bash
./build-cpu/lfm25-cpu-run \
  --model ./model/LFM2.5-230M \
  --prompt "Explique CUDA em uma frase." \
  --cpu-isa auto \
  --cpu-kv-cache bf16 \
  --cpu-kv-page-tokens 32 \
  --cpu-prefill-chunk 256 \
  --cpu-affinity compact \
  --threads 8 \
  --max-new-tokens 32
```

Standalone execution (1.2B-Instruct):

```bash
./build-cpu/lfm25-cpu-run \
  --model ./model/LFM2.5-1.2B-Instruct \
  --prompt "Explique CUDA em uma frase." \
  --cpu-isa auto \
  --cpu-kv-cache bf16 \
  --threads 8 \
  --max-new-tokens 32
```

Downloading a checkpoint:

```bash
# 230M (default)
./scripts/download_model.sh 230m
# 1.2B-Instruct
./scripts/download_model.sh 1.2b-instruct
```

Concurrent benchmark:

```bash
./build-cpu/lfm25-cpu-concurrent-benchmark \
  ./model/LFM2.5-1.2B-Instruct \
  "Explique como a CPU executa uma rede neural." \
  8 32 8 bf16 auto
```

Long-prompt chunk/page sweep:

```bash
MODEL=./model/LFM2.5-1.2B-Instruct/model.safetensors \
TOKENS=1024 KV=bf16 \
./scripts/cpu_long_prefill_benchmark.sh
```

## CUDA backend retained

The CUDA path continues to provide quantized weights, paged KV, prefix reuse,
ragged packed prefill, packed decode, continuous scheduling, CUDA Graphs and
cuBLAS/cuBLASLt. When `nvcc` is unavailable, CMake builds only the CPU targets.

## Validation helpers

```bash
./scripts/cpu_build_test.sh
./scripts/cpu_sanitizer_test.sh
./scripts/cpu_vnni_check.sh
./scripts/host_check.sh
./scripts/architecture_check.sh
```

The original checkpoints are not bundled. The packaging environment could not
run the complete official models, so the package does not claim end-to-end
logit parity, text quality or model tokens/s. See `VALIDATION.md`.
