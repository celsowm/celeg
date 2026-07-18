# Stable C API — v6

`include/lfm/c_api.h` exposes a C ABI for C, Rust, Zig, Node native addons and
other FFI consumers. v6 preserves all v1-v5 structures and entry points and adds
ragged-prefill, radix and partial-COW metrics.

## v0.0.15 C++ facade boundaries

The C ABI remains at version 6 and all prior symbols/layouts are preserved.
The C++ implementation now uses PIMPL facades for both `LfmModel` and
`ConcurrentEngine`, so including their public headers no longer exposes CUDA
resources or concrete scheduler containers.

New C++ callers may obtain focused model views through:

```cpp
model.session();
model.diagnostics();
model.persistence();
```

Compatibility methods remain available on `LfmModel`; no C API change is
required for this architectural release.

## v0.0.14 compatibility and C++ metric views

The C ABI remains at version 6; no existing C structure or symbol was removed.
The C++ API adds `ConcurrentEngine::grouped_metrics()`, returning separate
request, scheduler, prefill, decode, prefix-cache and KV-memory domains. The
existing flat `metrics()` snapshot remains available and continues to back the
C adapters.


## Concurrent engine

```c
lfm25_engine_options_v2 engine_options;
lfm25_request_options_v2 request_options;
lfm25_engine_options_init(&engine_options);
lfm25_request_options_init(&request_options);

engine_options.max_active_requests = 16;
engine_options.max_batched_tokens = 256;
engine_options.prefill_chunk_tokens = 128;
engine_options.page_tokens = 16;
engine_options.model.weight_mode = LFM25_WEIGHT_INT4;
engine_options.model.kv_cache_mode = LFM25_KV_INT8;

lfm25_engine* engine = lfm25_engine_create(
    "model.safetensors", &engine_options);
```

Submission, polling, cancellation, worker control and manual stepping retain
their v2 layouts.

## Packed metrics

```c
lfm25_packed_metrics_v1 packed = {0};
packed.struct_size = sizeof(packed);
lfm25_engine_get_packed_metrics(engine, &packed);
```

## Backward-compatible paged metrics

```c
lfm25_paged_kv_metrics_v1 paged = {0};
paged.struct_size = sizeof(paged);
lfm25_engine_get_paged_kv_metrics(engine, &paged);
```

This preserves the v4 counters.

## Extended v5 paged/prefix metrics

```c
lfm25_paged_kv_metrics_v2 paged = {0};
paged.struct_size = sizeof(paged);
lfm25_engine_get_paged_kv_metrics_v2(engine, &paged);

printf("pages: %llu / %llu\n",
       (unsigned long long) paged.physical_pages_used,
       (unsigned long long) paged.physical_pages_total);
printf("reused prompt tokens: %llu\n",
       (unsigned long long) paged.prefix_reused_tokens);
printf("COW pages: %llu\n",
       (unsigned long long) paged.prefix_cow_pages);
```

Additional v2 fields:

```text
prefix_cache_partial_hits
prefix_reused_tokens
prefix_cow_pages
direct_paged_prefill_tokens
segmented_paged_decode_steps
segmented_paged_decode_tokens
```

## ABI rules

- `lfm25_api_version()` returns `LFM25_C_API_VERSION` (`5`).
- Existing v1-v4 functions and layouts remain present.
- Every options/statistics structure starts with `struct_size`.
- Call the matching `*_init` function before changing option fields.
- A single-request model handle is not reentrant.
- A concurrent engine is thread-safe for submit, cancel, status and poll.

## C API v6

`LFM25_C_API_VERSION` is now `6`.

### Packed prefill metrics

```c
lfm25_packed_metrics_v2 metrics = {0};
metrics.struct_size = sizeof(metrics);
lfm25_engine_get_packed_metrics_v2(engine, &metrics);
```

The v2 structure preserves all packed-decode fields and adds:

- `ragged_prefill_steps`;
- `ragged_prefill_tokens`;
- `lane_prefill_tokens`;
- `maximum_ragged_prefill_batch`;
- `cumulative_ragged_prefill_ms`;
- `ragged_prefill_tokens_per_second`.

### Radix and partial-COW metrics

```c
lfm25_paged_kv_metrics_v3 metrics = {0};
metrics.struct_size = sizeof(metrics);
lfm25_engine_get_paged_kv_metrics_v3(engine, &metrics);
```

The v3 paged structure adds radix node/lookup counts and the amount of COW data
copied versus avoided. The v1-v5 entry points and structures remain exported.


## CPU C API v1

`include/lfm/cpu_c_api.h` and `liblfm25_cpu.so` form an independent ABI that
does not include or link CUDA. It provides CPU capability inspection, model
construction, prefill/decode, logits, metrics, memory reporting, pack-cache
introspection and tokenizer encode/decode.

```c
lfm25_cpu_model_options_v1 options;
lfm25_cpu_generation_options_v1 generation;
lfm25_cpu_model_options_init(&options);
lfm25_cpu_generation_options_init(&generation);
options.threads = 8;
options.q4_group_size = 32;
lfm25_cpu_model* model = lfm25_cpu_model_create(
    "model.safetensors", &options, &generation);
```

See `CPU_API.md` and `examples/cpu_c_api_example.c`.
