# CPU continuous batching

## Scheduler

`CpuConcurrentEngine` supports manual stepping or an internal worker. The
scheduler serializes execution over one persistent thread pool and combines:

1. packed decode with `M = active decode requests`;
2. ragged prefill waves for multiple prompts;
3. layer-major chunks when one long prompt remains;
4. decode-first token budgeting to protect inter-token latency.

Requests move through:

```text
Queued -> Prefilling -> Decoding -> Completed
                       |
                       +-> Cancelled / Failed
```

Cancellation is thread-safe. Results completing after cancellation are dropped
before commit.

## Prefix reuse

The v0.0.20 engine owns a radix-indexed `CpuPrefixCacheManager`. A hit restores:

- referenced physical KV pages for all six attention layers;
- all eight ShortConv states;
- final prompt logits;
- seen-token state and position.

It never restores RNG, seed or request-specific sampling settings. Longest-prefix
hits resume prefill at the first unmatched token. A shared partial tail page is
cloned before the session appends K/V, so cached snapshots remain immutable.
Eviction is LRU and constrained by both entry count and charged resident bytes.

## NUMA placement

With `CpuNumaMode::Local`, admission assigns requests to detected NUMA nodes in
round-robin order. KV pages are allocated from the corresponding arena and Linux
attempts best-effort `mbind`. Failure to bind is recorded but does not affect
correctness. Replicated per-node weights are deliberately reserved for a later
backend.

## Attention scheduling

Short contexts use the sequential online-softmax path. At or above the configured
threshold, attention creates tasks over query heads and page tiles. Each task
produces a partial maximum, denominator and FP32 value accumulator; partials are
combined with a stable online-softmax reduction.

## Metrics

The engine reports request lifecycle, scheduler time, prefill/decode throughput,
TTFT, ITL, maximum batches, parallel-attention calls, prefix hit/miss/partial-hit
counts, reused tokens, COW pages/bytes, evictions and NUMA page-binding outcomes.
