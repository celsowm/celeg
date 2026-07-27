# Concurrent inference runtime — v0.0.15

The engine combines continuous scheduling, chunked direct-paged prefill, packed
decode, physical KV pages and longest-prefix reuse.

## v0.0.15 engine decomposition

The public `ConcurrentEngine` is now a PIMPL facade. Its header contains only
stable options/results and no request containers, synchronization primitives,
CUDA page arena or packed executor.

Internal serving responsibilities are separated as follows:

```text
RequestRegistry  owns request records, IDs and the admission queue
BatchPlanner     selects admission and orders prefill/decode by priority
EngineWorker     owns the background thread and wake/stop lifecycle
Engine Impl      orchestrates model, KV, prefix cache and metrics
```

`RequestRegistry`, `BatchPlanner` and `EngineWorker` are part of the host-only
target and have independent unit tests. Manual `step()` and worker-thread mode
use the same engine orchestration path.

## v0.0.14 scheduler/cache boundary

`ConcurrentEngine` no longer implements radix lookup, LRU eviction, prefix page
ownership or partial-page COW directly. Those responsibilities live in
`PrefixCacheManager`. The manager depends only on `IKvPageAllocator`; production
uses `PhysicalPagedKvCache`, while host tests use a fake allocator.

The flat `ConcurrentMetrics` snapshot remains compatible. C++ clients may use
`grouped_metrics()` for request, scheduler, prefill, decode, prefix and KV views.


## Scheduler iteration

1. Admit queued requests when a lane and sufficient physical pages exist.
2. Find the longest compatible cached prefix.
3. Retain its full pages and COW-clone a partial final page when necessary.
4. Spend the remaining token budget on direct-to-page prompt chunks.
5. Run compatible ready requests through one packed model pass.
6. Use segmented paged attention automatically for long contexts.
7. Publish tokens, finish/cancel requests and release page references.

## Mutable request state

Each request keeps independent:

- position and generated-token count;
- sampling configuration and RNG;
- seen-token bitmap;
- logits;
- eight ShortConv states;
- physical KV page table;
- output queue and cancellation state.

Model weights and the physical KV arena are shared.

## Prefix reuse

The cache accepts arbitrary prompt lengths. Exact hits skip all prompt work;
partial hits process only the uncached suffix. Entries are bounded by
`prefix_cache_entries`, use LRU eviction and can also be evicted under physical
page pressure.

C++ tuning:

```cpp
lfm::ConcurrentEngineOptions options;
options.page_tokens = 16;
options.logical_kv_pages = 0;      // derive capacity
options.prefix_cache = true;
options.prefix_cache_entries = 64;
options.packed_decode = true;
options.packed_min_batch = 1;
```

C API v5 adds `lfm25_paged_kv_metrics_v2` while preserving the older metrics
structure and all v1-v4 entry points.

## Scheduler policies

`GuaranteedNoEvict` reserves enough pages for prompt plus requested generation
before admission. `MaxUtilization` begins with a smaller reservation and grows
incrementally; a request can pause while no page is free.

## Remaining performance work

- true ragged packed prefill across several prompts;
- block-hash/radix prefix indexing instead of bounded linear scan;
- CUDA Graph buckets for the complete dynamic scheduler iteration;
- benchmark-guided chunk and page-size autotuning;
- asynchronous page clone on a dedicated transfer stream.

## Flattened ragged packed prefill

The scheduler flattens all selected prompt chunks into one token batch. Every
token carries its request position and a copy of that request's physical page
table. `PackedDecodeExecutor::prefill` executes the shared weights once with
`M = total prompt tokens`.

Each request retains an independent causal KV history and ShortConv ring state;
the ShortConv kernel advances each request span in token order. Decode tokens
keep a reserved share of `max_batched_tokens`; prefill consumes only the
remaining budget.

There are no internal dependency-safe token waves: one admitted ragged chunk
causes one transformer pass. BF16/INT8 paged KV and segmented attention use the
same flattened token mapping.

New C++ controls:

```cpp
engine_options.ragged_packed_prefill = true;
engine_options.ragged_prefill_min_batch = 2;
```

New counters include `ragged_prefill_steps`, `ragged_prefill_tokens`,
`lane_prefill_tokens`, `maximum_ragged_prefill_batch` and
`cumulative_ragged_prefill_ms`. `PackedDecodeMetrics` additionally exposes
`ragged_prefill_transformer_passes` for structural verification.
