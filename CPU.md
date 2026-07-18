# Native CPU backend — v0.0.20

The CPU runtime executes the LFM2.5-230M topology directly from the original
`safetensors`. Groupwise-Q4 weights and one persistent thread pool are shared by
all sessions. CUDA is optional and is not linked into `liblfm25_cpu.so`.

## Long-prompt prefill

Prompts above the configured threshold use layer-major chunks. QKV, attention
output, ShortConv projections and both MLP projections run with
`M = chunk_tokens`. Only causal attention and the three-tap ShortConv recurrence
remain ordered by token position.

```bash
--cpu-prefill-chunk 256
--cpu-prefill-threshold 16
```

## Physically paged KV

Each attention layer owns one global FP32/BF16 page arena. Sessions hold page
IDs rather than contiguous K/V vectors. The default page contains 32 tokens.

```bash
--cpu-kv-cache bf16
--cpu-kv-page-tokens 32
```

Pages are aligned, reference counted and recycled. Prefill and decode write
directly to pages; no contiguous staging cache is created.

## Tile-parallel paged attention

Short contexts retain the low-overhead one-pass online-softmax path. Long
contexts are split into tasks by `query head × page tile`. Each task emits a
partial `(maximum, denominator, weighted-value accumulator)` state. Partial
states are merged with numerically stable online-softmax composition.

```bash
--cpu-attention-threshold 256
--cpu-attention-page-tile 4
```

The implementation uses the existing persistent thread pool; it never creates
threads from inside an attention call. Scores, softmax and accumulators remain
FP32 for FP32 and BF16 KV storage.

## Shared radix prefix cache

The concurrent engine indexes complete prompt snapshots by token ID in a radix
tree and restores the longest cached prefix. A snapshot contains:

- all six attention page tables;
- the eight ShortConv recurrent states;
- final prefix logits;
- seen-token state;
- position and NUMA placement metadata.

RNG seed and sampling configuration are intentionally not shared.

```cpp
CpuConcurrentEngineOptions options;
options.prefix_cache = true;
options.prefix_cache_max_entries = 256;
options.prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;
```

The cache uses LRU eviction by entry and byte limits. Metrics include exact and
partial hits, reused tokens, COW pages and copied bytes.

## Partial-page copy-on-write

Full pages can remain shared. When a cached prefix ends in the middle of a page,
the receiving request clones only the valid K/V token interval before appending
new tokens. Cache insertion also creates an immutable private tail page so the
request that populated the cache cannot mutate it later.

This applies independently to every attention layer and supports FP32 and BF16.

## NUMA-local KV pages

```bash
--cpu-numa disabled|local
```

In `local` mode, requests are distributed across detected NUMA nodes and new KV
pages are allocated with the request's preferred node. On Linux the runtime
attempts `mbind`; if the host or container denies it, allocation remains valid
and the failure is reported in page diagnostics. `replicate-weights` is reserved
for a later release and is rejected rather than silently emulated.

## Validation against the official model

Python is used only to export an offline reference:

```bash
python scripts/export_cpu_reference.py \
  --model LiquidAI/LFM2.5-230M \
  --prompt "Explique CUDA." \
  --output reference/cuda
```

Then compare with the standalone C++ runtime:

```bash
./build-cpu/lfm25-cpu-compare-reference \
  ./model/LFM2.5-230M/model.safetensors \
  reference/cuda fp32 scalar 1
```

The comparator reports maximum/mean absolute error, RMSE, cosine similarity,
top-1 equality and top-10 intersection.

## Prefix-cache benchmark

```bash
./build-cpu/lfm25-cpu-prefix-cache-benchmark \
  ./model/LFM2.5-230M \
  "Explique CUDA." 4 \
  " Agora explique memória compartilhada."
```

The first request populates the cache; later requests exercise exact or longest
prefix reuse.

## ISA paths

| ISA | Status |
|---|---|
| scalar | executable |
| AVX2/FMA | executable |
| AVX-VNNI | executable |
| AVX-512 VNNI | executable |
| NEON | executable, not physically tested here |
| AMX/I8MM/SME2 | detected only |

## Remaining limits

- Page tiles currently parallelize generic FP32 accumulation rather than using
  an AVX-512/AMX attention-specific microkernel.
- NUMA-local page placement is implemented, but weights are not yet replicated
  per socket and the worker pool is not partitioned into node-local sub-pools.
- Prefix cache accounting conservatively charges the full referenced page bytes
  to each entry, even when full pages are shared by several cache entries.
- The official checkpoint was not available in the packaging environment, so
  the new reference tools were built but no official logits, text quality,
  TTFT or full-model tokens/s are claimed in this release.
