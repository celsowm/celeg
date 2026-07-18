# Physical paged KV and prefix reuse

## v0.0.15 ownership interface

Scheduler/cache policy sees KV memory through `IKvPageAllocator`. The interface
contains allocation, retain/release, partial-page cloning and capacity metrics,
but exposes no CUDA pointers. `PhysicalPagedKvCache` implements the interface
and separately retains device-specific accessors for packed kernels.

The obsolete contiguous-cache import from `LfmModel` was removed, eliminating
the friend relationship and circular public-header dependency.


## Pool layout

One `PhysicalPagedKvCache` is created per `ConcurrentEngine`. Pages are shared
by active requests and cached prefixes.

```text
BF16:
key/value[page][6 attention slots][page token][8 KV heads][64]

INT8:
key/value[page][6 attention slots][page token][8 KV heads][64]
scale[page][6 attention slots][page token][8 KV heads]
```

Attention slots map model layers `2, 4, 6, 8, 10, 12` to `0..5`.

A request owns one reference to every page in its table. A prefix-cache entry
owns its own references. Pages return to the free list only when their reference
count reaches zero.

## Direct paged prefill

Concurrent models can disable their private contiguous KV allocation. Every
prompt token:

1. performs embedding and the 14 model layers;
2. writes K/V directly at `page_table[position / page_tokens]`;
3. reads previous keys and values through the same page table;
4. updates request-local ShortConv state;
5. computes logits only at the final prompt token.

Chunked scheduling changes only when the loop yields to other requests. It does
not copy or import KV between chunks.

## Longest-prefix reuse

The cache key still uses a 64-bit token hash plus full token equality for exact
entry identity. Admission searches the bounded cache and selects the longest
cached token vector that is a prefix of the incoming prompt.

A hit restores:

- logits at the cached position;
- the vocabulary seen-token bitmap;
- all eight ShortConv recurrent states;
- prompt position.

The receiving request initializes its own RNG from its configured seed. When
the incoming prompt is longer, direct paged prefill processes only the suffix.

## Partial-page copy-on-write

A cached prefix may end anywhere inside a physical page.

Two COW points keep cached data immutable:

- **cache insertion:** when the creator's prompt ends mid-page, the cache gets a
  cloned final page before the creator begins generation;
- **cache hit:** a continuing request clones the cached partial final page before
  suffix prefill or generation.

Full pages remain reference-counted and shared. The clone copies K/V for all six
attention layers and, for INT8 mode, all scale vectors.

## Segmented paged attention

For long contexts, paged GQA divides each `(request, query head)` into chunks.
Each chunk produces:

- local maximum score;
- local softmax denominator;
- local weighted-value accumulator.

A reduction kernel combines partials with stable online-softmax rescaling.
Both BF16 and INT8 page layouts are supported. Automatic selection follows
`attention_auto_threshold`; chunk width follows `attention_chunk_tokens`.

## Allocator guarantees

Reference operations are transactional, including lists with duplicate page
IDs. Invalid retains/releases cannot partially mutate earlier pages. The pool
rejects invalid IDs, double release, reference overflow and size overflow.

## v0.0.13 radix index and token-range COW

The prefix cache now maintains a radix tree keyed by token IDs. Longest-prefix
lookup follows the incoming token sequence once and returns the deepest cached
terminal entry. LRU ownership remains in the cache records; eviction removes the
same entry from the radix tree transactionally.

For a partial final page, `clone_page_prefix(source, used_tokens)` copies each
attention layer's initialized `[used_tokens, kv_width]` interval. INT8 mode also
copies only the corresponding scale intervals. The uninitialized page suffix is
not copied and is overwritten before its position can be attended to.

The physical page allocation unit remains unchanged, so this optimization
reduces device-to-device copy traffic rather than page capacity. Metrics expose
`prefix_cow_bytes_copied` and `prefix_cow_bytes_saved`.
