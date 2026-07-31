# Benchmark guide

## Built-in decode benchmark

```bash
./build/celeg-run \
  --model ./model/LFM2.5-230M \
  --prompt "Explique Tensor Cores." \
  --context 8192 \
  --benchmark-warmup 16 \
  --benchmark-decode 128 \
  --memory-report
```

Prefill uses wall-clock timing after the call returns. Decode uses CUDA events
around a sequence of asynchronous graph launches or direct decode steps. No
token is copied to the host inside the measured loop.

## Comparison matrix

```bash
MODEL=./model/LFM2.5-230M \
CONTEXT=8192 \
DECODE_TOKENS=128 \
WARMUP_TOKENS=16 \
./scripts/benchmark.sh
```

The matrix includes:

- strict cuBLAS with legacy sampling;
- fused sampling;
- fast BF16 with cuBLAS;
- cached and autotuned cuBLASLt;
- W8A16;
- W8A16 segmented attention;
- W4A16;
- W4A16 automatic attention.

## Automatic attention measurements

`--attention-mode auto` chooses the graph by host-side context position. The
benchmark determines whether its complete warmup and measured interval uses the
single graph, the segmented graph, or crosses the threshold. Every required
graph is captured before event timing begins.

Example:

```bash
./build/celeg-run \
  --model ./model/LFM2.5-230M \
  --prompt "Explique CUDA." \
  --context 16384 \
  --fast-attention \
  --attention-mode auto \
  --attention-auto-threshold 4096 \
  --attention-chunk-tokens 256 \
  --benchmark-warmup 16 \
  --benchmark-decode 256
```

The optimal threshold is hardware- and context-dependent. Measure at multiple
starting prompt lengths; do not infer the crossover only from `max_context`.

## Quantized modes

Report both latency and quality. A faster mode that changes task output is not
a drop-in replacement.

```bash
MODEL=./model/LFM2.5-230M ./scripts/compare_weight_modes.sh
```

Useful measurements include:

- weight and total VRAM;
- prefill tokens/s;
- decode tokens/s at several context lengths;
- top-1 and top-10 logit agreement;
- RMSE and maximum logit error;
- greedy sequence agreement;
- task-level evaluation for the intended prompts.

The current W8A16/W4A16 kernels are warp-oriented CUDA implementations, not
native low-bit tensor-core MMA kernels. Expect decode and prefill behavior to
differ.

## v0.0.8 additions

`--runtime-metrics` reports end-to-end wall-clock values visible to the caller:

```text
runtime.prefill_tokens
runtime.prefill_ms
runtime.prefill_tokens_per_second
runtime.decode_tokens
runtime.decode_ms
runtime.decode_tokens_per_second
```

Unlike the CUDA-event benchmark, decode wall time includes the one-token D2H
copy and stream synchronization performed by `decode()`.

The benchmark matrix now includes packed INT4 weights with INT8 KV cache. Use
`compare_kv_modes.sh` to compare first-token logits and reported cache memory
between BF16 and INT8 KV modes.

## v0.0.10 packed concurrent benchmark

```bash
./build/celeg-concurrent-benchmark \
  ./model/LFM2.5-230M/model.safetensors \
  ./model/LFM2.5-230M/tokenizer.json \
  "Explique CUDA." 16 64
```

Or:

```bash
MODEL_DIR=./model/LFM2.5-230M \
REQUESTS=16 MAX_NEW=64 \
./scripts/concurrency_benchmark.sh
```

Reported values now include:

- aggregate decode tokens per second;
- average TTFT and ITL;
- packed decode steps and tokens;
- lane-fallback tokens and fallback batch count;
- maximum packed batch reached;
- packed-path tokens per second;
- scheduler iterations and logical page usage.

Compare packed and lane paths at identical weight mode, KV mode, context,
sampling and request population. A physical GPU run is required before drawing
performance conclusions; this package does not contain measured v0.0.10 speedup.

## v0.0.11 physical-page and prefix-cache counters

`celeg-concurrent-benchmark` additionally prints:

```text
physical_pages_current
physical_pages_total
physical_kv_bytes
prefix_cache_hits
prefix_cache_misses
prefix_cache_inserts
prefix_cache_evictions
```

In v0.0.11, reuse required identical page-aligned prompts. v0.0.12 removes
that boundary restriction and can also reuse the longest cached prefix. These
counters are instrumentation only; no measured GPU result is bundled with this
package.

### Dedicated prefix-cache benchmark

```bash
./build/celeg-prefix-cache-benchmark \
  ./model/LFM2.5-230M/model.safetensors \
  ./model/LFM2.5-230M/tokenizer.json \
  "Explique CUDA." 4 1
```

The utility runs identical requests sequentially so the first request can
populate the cache before later submissions. The v0.0.12 utility intentionally
does not pad the prompt, allowing partial-page COW instrumentation.

## v0.0.12 direct-page and longest-prefix counters

The concurrent benchmark now also reports:

```text
direct_paged_prefill_tokens
prefix_cache_partial_hits
prefix_reused_tokens
prefix_cow_pages
segmented_paged_decode_steps
segmented_paged_decode_tokens
```

`direct_paged_prefill_tokens` should equal concurrent prompt tokens processed
without a cache hit. `prefix_reused_tokens` counts prompt tokens skipped by
exact or partial hits. `prefix_cow_pages` includes immutable partial-page clones
created during cache insertion and later reuse.

The dedicated prefix benchmark no longer pads the prompt to a page boundary.
This deliberately exercises arbitrary-length prefix insertion and partial-page
copy-on-write:

```bash
./build/celeg-prefix-cache-benchmark \
  ./model/LFM2.5-230M/model.safetensors \
  ./model/LFM2.5-230M/tokenizer.json \
  "Explique CUDA." 4 1 " Agora detalhe memória compartilhada."
```

The optional suffix is omitted on the first iteration and appended on later
iterations, exercising continuation from the longest cached prefix. Without a
suffix, all iterations exercise exact hits.
A physical GPU run is required to compare output parity and measure the cost of
page cloning or segmented attention.

## v0.0.13 ragged-prefill counters

The concurrent benchmark now reports whether prompt work used the packed
wavefront or lane path:

```text
ragged_prefill_steps
ragged_prefill_tokens
lane_prefill_tokens
maximum_ragged_prefill_batch
ragged_prefill_tokens_per_second
```

It also reports radix lookup activity and partial-page COW traffic. To exercise a
large prefill batch, submit several requests before entering the manual `step()`
loop, as `celeg-concurrent-benchmark` already does.


## CPU benchmarks — v0.0.18

Primitive Q4 GEMV/GEMM benchmark:

```bash
./build-cpu/celeg-cpu-kernel-benchmark \
  4096 1024 20 8 32 1 avx512-vnni compact
```

Positional arguments:

```text
output_rows input_columns iterations threads q4_group batch isa affinity
```

For `batch=1`, the benchmark calls GEMV. For `batch>1`, it calls the packed
`M>1` path and reports both calls/s and batch rows/s. Effective bandwidth counts
packed Q4 weights and BF16 group scales only.

On the packaging host, a 4096×1024 group-32 synthetic matrix with eight threads
measured approximately:

```text
AVX2/FMA, M=1          2.5 GB/s effective weight bandwidth
AVX-512 VNNI, M=1    12.0 GB/s effective weight bandwidth
AVX-512 VNNI, M=8    25.1 GB/s effective weight bandwidth
```

These are microkernel measurements on one virtualized host and are not model
throughput claims.

Model benchmark, after supplying the checkpoint:

```bash
BUILD_DIR=./build-cpu \
MODEL_DIR=./model/LFM2.5-230M \
THREADS=8 GROUP=32 TOKENS=32 AFFINITY=compact \
./scripts/cpu_benchmark.sh
```

Do not compare the synthetic linear number directly with end-to-end model
tokens/s. The model also executes normalization, attention, ShortConv, RoPE,
SwiGLU, sampling and additional memory traffic.

## v0.0.18 CPU continuous batching

Run one population:

```bash
./build-cpu/celeg-cpu-concurrent-benchmark \
  ./model/LFM2.5-230M \
  "Explique como a CPU executa uma rede neural." \
  8 32 8 bf16 auto
```

Arguments after the prompt are requests, maximum generated tokens per request,
thread count, KV mode and ISA.

Run the standard concurrency matrix:

```bash
MODEL=./model/LFM2.5-230M \
THREADS=8 KV=bf16 ISA=auto MAX_NEW=32 \
./scripts/cpu_concurrency_benchmark.sh
```

For every population, record:

- aggregate generated tokens divided by complete benchmark wall time;
- packed decode and ragged prefill kernel throughput;
- average TTFT and ITL;
- maximum observed prefill and decode batch;
- peak RSS and total CPU utilization externally;
- output/logit parity against standalone execution.

Compare `KV=fp32` and `KV=bf16` separately. BF16 reduces cache bytes but may
change logits; performance and quality must both be measured on the target CPU.

## v0.0.20 long-prompt CPU prefill

Run one configuration:

```bash
./build-cpu/celeg-cpu-prefill-benchmark \
  ./model/LFM2.5-230M/model.safetensors \
  1024 256 32 bf16 auto 8
```

Arguments are `model`, prompt token count, chunk tokens, page tokens, KV mode,
ISA and thread count. The utility reports prefill time/throughput and physical
page memory.

Sweep page and chunk sizes:

```bash
MODEL=./model/LFM2.5-230M/model.safetensors \
TOKENS=1024 KV=bf16 THREADS=8 \
./scripts/cpu_long_prefill_benchmark.sh
```

Compare at least chunks 64/128/256/512 and pages 16/32/64. Measure TTFT and RSS
with the official checkpoint before changing defaults.

## v0.0.20 CPU prefix reuse

Run repeated identical prompts:

```bash
./build-cpu/celeg-cpu-prefix-cache-benchmark \
  ./model/LFM2.5-230M/model.safetensors \
  4 1 "Explique CUDA." \
  " Agora detalhe memória compartilhada."
```

The first request populates the cache. Later requests either reuse the full
prompt or restore the longest common prefix and process only the suffix. Record:

```text
prefix_cache_hits
prefix_cache_partial_hits
prefix_reused_tokens
prefix_cow_pages
prefix_cow_bytes
prefix_cache_evictions
```

Prefix snapshots include the six attention page tables, the eight ShortConv
states, final logits and seen-token state. RNG and request-specific sampling
configuration remain private.

## v0.0.20 official-reference comparison

The Python tool is validation-only and is not a runtime dependency:

```bash
python3 scripts/export_cpu_reference.py \
  --model ./model/LFM2.5-230M \
  --prompt "Explique CUDA em uma frase." \
  --output ./validation/cpu-reference
```

Compare the native CPU backend:

```bash
./build-cpu/celeg-cpu-compare-reference \
  ./model/LFM2.5-230M/model.safetensors \
  ./validation/cpu-reference \
  bf16 auto 8
```

The comparator reports maximum and mean absolute error, RMSE, cosine
similarity, top-1 agreement and top-10 intersection. Run the matrix for scalar,
AVX2, VNNI, FP32/BF16 KV, incremental/chunked prefill, page boundaries and
prefix-cache/COW paths before claiming numerical parity.

## v0.0.20 NUMA and parallel-attention measurements

On multi-socket Linux hosts, compare `--cpu-numa disabled` and
`--cpu-numa local`. Page binding uses a best-effort `mbind`; report bind failures
and verify placement with operating-system tools rather than assuming success.
For attention, sweep the parallel threshold and page tile size while measuring
TTFT, ITL P50/P95/P99 and CPU utilization. The sequential path should remain the
baseline for short contexts because task scheduling can cost more than it saves.
