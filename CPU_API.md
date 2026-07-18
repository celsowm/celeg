# CPU C API v5

`liblfm25_cpu.so` remains independent of CUDA. ABI v5 preserves all v1-v4
symbols and adds controls for tile-parallel attention and NUMA placement.

## Model options

```c
lfm25_cpu_model_options_v5 options;
lfm25_cpu_model_options_v5_init(&options);

options.kv_cache_mode = LFM25_CPU_KV_BF16;
options.kv_page_tokens = 32;
options.prefill_chunk_tokens = 256;
options.prefill_chunk_threshold = 16;
options.attention_parallel_threshold = 256;
options.attention_page_tile = 4;
options.numa_mode = LFM25_CPU_NUMA_LOCAL;

lfm25_cpu_model* model = lfm25_cpu_model_create_v5(
    "model.safetensors", &options, NULL);
```

`LFM25_CPU_NUMA_REPLICATE_WEIGHTS` is reserved and currently rejected.

## Concurrent engine and prefix cache

```c
lfm25_cpu_engine_options_v3 engine_options;
lfm25_cpu_engine_options_v3_init(&engine_options);

engine_options.prefix_cache = 1;
engine_options.prefix_cache_max_entries = 256;
engine_options.prefix_cache_max_bytes = 512ULL * 1024ULL * 1024ULL;

lfm25_cpu_engine* engine = lfm25_cpu_engine_create_v3(
    "model.safetensors", &options, &engine_options);
```

## Metrics v3

```c
lfm25_cpu_engine_metrics_v3 metrics = {0};
metrics.struct_size = sizeof(metrics);
lfm25_cpu_engine_get_metrics_v3(engine, &metrics);
```

Additional counters include prefix hits/misses/partial hits, reused tokens,
evictions, COW pages/bytes and parallel-attention calls.
