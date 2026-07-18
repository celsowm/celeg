# lfm25-native-cpp

Experimental C++20 inference runtime specialized for
`LiquidAI/LFM2.5-230M`, with independent NVIDIA CUDA and native CPU backends.
The runtime reads the original `safetensors`, `config.json` and
`tokenizer.json` files. Python or another model-serving runtime is not required.

## v0.0.20: parallel paged attention and shared CPU prefixes

The native CPU serving path now adds the missing long-context and reuse pieces:

- adaptive paged GQA parallelism over query heads and page tiles;
- numerically stable reduction of partial online-softmax states;
- radix-indexed longest-prefix cache over token IDs;
- complete hybrid prefix snapshots: paged K/V, all ShortConv states, final
  logits and seen-token state;
- partial-page copy-on-write that copies only initialized K/V tokens;
- LRU eviction constrained by entry count and resident-byte budget;
- NUMA-aware request placement and best-effort page binding on Linux;
- explicit metrics for prefix hits, reused tokens, COW traffic, page placement
  and parallel-attention calls;
- validation-only Transformers exporter and native C++ logit comparator;
- CPU C API v5 while preserving the previous CPU API entry points.

Long-prompt layer-major prefill, packed decode, ragged multi-request prefill,
Q4 groupwise weights and FP32/BF16 physical KV pages remain available. The CPU
microkernels cover scalar, AVX2/FMA, AVX-VNNI, AVX-512 VNNI and NEON. See
`CPU.md`, `CPU_CONCURRENCY.md`, `CPU_API.md` and `BENCHMARK.md`.

## CPU-only build

```bash
cmake -S . -B build-cpu \
  -DLFM_ENABLE_CUDA=OFF \
  -DLFM_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

Standalone execution:

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

Concurrent benchmark:

```bash
./build-cpu/lfm25-cpu-concurrent-benchmark \
  ./model/LFM2.5-230M \
  "Explique como a CPU executa uma rede neural." \
  8 32 8 bf16 auto
```

Long-prompt chunk/page sweep:

```bash
MODEL=./model/LFM2.5-230M/model.safetensors \
TOKENS=1024 KV=bf16 \
./scripts/cpu_long_prefill_benchmark.sh
```

Or run the 1/2/4/8/16-request matrix:

```bash
MODEL=./model/LFM2.5-230M \
THREADS=8 KV=bf16 \
./scripts/cpu_concurrency_benchmark.sh
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

The original checkpoint is not included. The packaging environment could not
run the complete official model, so the package does not claim end-to-end
logit parity, text quality or model tokens/s. See `VALIDATION.md`.
