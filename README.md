# celeg-native-cpp

Experimental C++20 inference runtime for LiquidAI LFM2.5 checkpoints, with
independent NVIDIA CUDA and native CPU backends. The runtime reads the
original `safetensors`, `config.json` and `tokenizer.json` files. Python or
another model-serving runtime is not required.

## Supported variants

| Variant id              | HuggingFace repo                       | Hidden | Layers | Q heads | KV heads | Head dim | Vocab  |
|-------------------------|----------------------------------------|-------:|-------:|--------:|---------:|---------:|-------:|
| `lfm2.5-230m`           | `LiquidAI/LFM2.5-230M`                 |   1024 |     14 |      16 |        8 |       64 |  65536 |
| `lfm2.5-1.2b-instruct`  | `LiquidAI/LFM2.5-1.2B-Instruct`        |   2048 |     16 |      32 |        8 |       64 |  65536 |
| `lfm2.5-1.2b-thinking`  | `LiquidAI/LFM2.5-1.2B-Thinking`        |   2048 |     16 |      32 |        8 |       64 |  65536 |

Variants are selected at runtime from the checkpoint's `config.json` through
`celeg::ModelVariantRegistry`. Adding a new variant does not require editing the
kernels: register a new `IModelVariant` subclass and the runtime will pick it
up automatically (Open/Closed Principle).

## v0.0.21: multi-variant support

The runtime is no longer specialized for the 230M checkpoint. The major
refactor is the introduction of `celeg::ModelShape`, a runtime topology
descriptor that replaces the former `CelegConfig` constexpr struct. Every buffer
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
- CUDA C API gains `celeg_model_vocab_size`;
- CMake `CELEG_VARIANTS` option lists which variants the build advertises
  (default: `230m;1.2b-instruct;1.2b-thinking`).

The CUDA backend continues to provide quantized weights, paged KV, prefix
reuse, ragged packed prefill, packed decode, continuous scheduling, CUDA
Graphs and cuBLAS/cuBLASLt. When `nvcc` is unavailable, CMake builds only the
CPU targets.

## Portable build and verification

The standard developer entrypoint works from Windows, Linux and agent
harnesses. It discovers Visual Studio, CUDA, the GPU architecture, runtime
libraries and cached checkpoints without machine-specific paths:

```text
python scripts/dev.py doctor
python scripts/dev.py verify
```

`auto` uses CUDA when a compatible toolkit and GPU architecture are available,
otherwise it builds the CPU backend. Select a backend explicitly when required:

```text
python scripts/dev.py verify --backend cpu
python scripts/dev.py verify --backend cuda
python scripts/dev.py doctor --backend cuda --json
```

Builds default to `RelWithDebInfo` under
`out/<platform>-<backend>-relwithdebinfo`. Common overrides include
`--build-type Release`, `--arch 86`, `--jobs 8` and `--build-dir PATH`.
`build`, `test`, `smoke` and `verify` always perform a fresh CMake configure so
an old cache cannot retain a different compiler or CUDA toolkit.

For IDEs and direct CMake use, equivalent CPU/CUDA Release and RelWithDebInfo
presets are available:

```text
cmake --preset cpu-relwithdebinfo
cmake --build --preset cpu-relwithdebinfo
ctest --preset cpu-relwithdebinfo
```

Standalone execution (230M):

```bash
./out/linux-cpu-relwithdebinfo/celeg-cpu-run \
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
./out/linux-cpu-relwithdebinfo/celeg-cpu-run \
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
# 1.2B-Thinking
./scripts/download_model.sh 1.2b-thinking
```

Concurrent benchmark:

```bash
./out/linux-cpu-relwithdebinfo/celeg-cpu-concurrent-benchmark \
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

On Windows, replace `linux` in those paths with `windows` and append `.exe`.

## CUDA backend

The CUDA path continues to provide quantized weights, paged KV, prefix reuse,
ragged packed prefill, packed decode, continuous scheduling, CUDA Graphs and
cuBLAS/cuBLASLt. Windows builds stage the selected toolkit's required runtime
DLLs beside the executables. A cached 230M checkpoint is exercised by
`scripts/dev.py smoke`; if it is absent, inference is clearly skipped and no
download occurs.

## Validation helpers

```bash
python scripts/dev.py verify
./scripts/cpu_sanitizer_test.sh
./scripts/cpu_vnni_check.sh
./scripts/host_check.sh
./scripts/architecture_check.sh
```

The original checkpoints are not bundled. The packaging environment could not
run the complete official models, so the package does not claim end-to-end
logit parity, text quality or model tokens/s. See `VALIDATION.md`.
