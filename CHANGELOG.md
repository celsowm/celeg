# Changelog

## Unreleased

- Added OpenAI-compatible tool definitions, tool-call DTOs, nullable assistant
  content, capability-aware request validation, `tool_calls` finish reasons,
  and structured escaped error responses.
- Added native LFM2 and Gemma 4 tool-call codecs. Their marker syntax follows
  the checked-in llama.cpp templates; Granite remains explicitly unsupported
  until its parser is verified against a checkpoint vocabulary.
- Documented the checkpoint-to-protocol architecture chain and added boundary
  rules for profile data, neutral checkpoint contracts, and serving isolation.

## v0.0.20

- Added adaptive tile-parallel paged GQA over `query head × page tile` tasks.
- Added stable reduction of partial online-softmax states without materializing an O(sequence) score vector.
- Added configurable `attention_parallel_threshold` and `attention_page_tile` runtime controls.
- Added CPU radix prefix cache with exact and longest-prefix lookup.
- Added complete hybrid-prefix snapshots: six KV page tables, eight ShortConv states, logits, seen-token state and position.
- Added partial-page copy-on-write for FP32/BF16 K/V during cache insertion, cache acquisition and subsequent request writes.
- Added LRU eviction with entry and byte limits plus prefix hit/reuse/COW metrics.
- Added NUMA topology discovery, per-request node assignment, aligned page storage and best-effort Linux `mbind` placement.
- Added CPU C API v5 model options, engine v3 options and engine metrics v3 while preserving older entry points.
- Added `celeg-cpu-prefix-cache-benchmark`.
- Added `export_cpu_reference.py`, `celeg-cpu-compare-reference` and a shell wrapper for official-logit parity checks.
- Added parallel-attention, prefix-cache/COW and NUMA unit tests; CPU CTest now contains 31 tests.
- Preserved layer-major long-prompt prefill, packed decode, ragged prefill and all previous CUDA sources.

## v0.0.18

- Added `CpuModel::clone_session()` with immutable Q4 weights and one persistent thread pool shared across sessions.
- Added `CpuPackedExecutor` for full-model packed token execution with `M = active batch` linear projections.
- Added ragged/wavefront packed prefill across unrelated prompts.
- Added packed multi-request decode and terminal-row-only LM-head execution.
- Added `CpuConcurrentEngine` with continuous admission, token budgeting, priority, cancellation, polling, manual stepping and an optional worker thread.
- Added FP32/BF16 CPU KV modes with FP32 attention accumulation.
- Changed CPU KV allocation to lazy sequence growth instead of reserving the maximum context at request admission.
- Added TTFT, ITL, prefill/decode throughput, observed-batch and request-lifecycle metrics.
- Added CPU C API v3 and concurrent engine functions while preserving v1/v2 symbols.
- Added the concurrent C example, benchmark utility and 1/2/4/8/16 request benchmark script.
- Added BF16 GQA equivalence, concurrent metrics and CPU option tests.
- Preserved all CUDA backend functionality.

## v0.0.17

- Added executable AVX-VNNI and AVX-512 VNNI Q4×Q8 dot-product kernels.
- Added per-group dynamic INT8 activation quantization with FP32 scales and precomputed group sums.
- Added the exact signed-Q4 correction used with unsigned-byte × signed-byte `VPDPBUSD`.
- Vectorized Q4 nibble unpacking into 32-byte VNNI tiles.
- Changed automatic CPU dispatch to select the highest executable kernel rather than diagnostic-only AMX/SME capabilities.
- Reworked `CpuLinearEngine::gemm()` to schedule the flattened batch-row × output-row space and reuse one Q8 quantization per input row.
- Added Linux process-affinity discovery, basic NUMA-node discovery, and compact/scatter worker pinning.
- Added CPU C API v2 with affinity options and topology diagnostics while preserving v1.
- Expanded the CPU benchmark with batch, explicit ISA, affinity, pinned-worker and effective-bandwidth reporting.
- Added Q8, VNNI equivalence, batched GEMM, topology and affinity tests.
- Preserved the standalone CPU model, `.lfmpack` format and CUDA backend.

## v0.0.15

- Converted `CelegModel` into a PIMPL facade and removed CUDA/cuBLAS/checkpoint implementation dependencies from its public header.
- Added focused `CelegInferenceSession`, `CelegDiagnostics` and `CelegPersistence` C++ views while retaining compatibility forwarding methods.
- Replaced the common boolean-discriminated `Layer` record with `std::variant<AttentionLayer, ConvolutionLayer>` and layer-specific state.
- Converted `ConcurrentEngine` into a PIMPL facade whose public header no longer exposes mutexes, queues, CUDA page arenas or packed executors.
- Extracted host-testable `RequestRegistry`, `BatchPlanner` and `EngineWorker` components.
- Routed admission and prefill/decode ordering through `BatchPlanner` instead of embedding priority sorting in the engine.
- Split packed decode/prefill into validation, metadata, embedding, QKV projection, paged/local attention, convolution, MLP and finalization stages.
- Reduced packed `decode()` to about 60 lines and `prefill_step()` to about 65 lines, with automated growth limits.
- Added registry, planner and worker lifecycle tests.
- Expanded architecture checks for public-header isolation, typed layer variants, engine PIMPL and packed stage sizes.
- Preserved C API v6 and all 43 exported C symbols.

## v0.0.14

- Added `runtime_types.hpp` to decouple model/runtime value types from CUDA model internals.
- Added immutable, validated `ExecutionPlan` selection for BF16/cuBLAS, BF16/cuBLASLt, W8A16, W4A16, sampling and segmented-attention policy.
- Added explicit `LinearStorageKind` invariants and rejected invalid pointer/scale combinations.
- Replaced independent ready/pending booleans with one `SessionPhase` state.
- Added `SharedModelWeights` as the explicit shared immutable checkpoint store, including serialized first-load initialization across concurrent sessions.
- Extracted radix lookup, LRU eviction, page ownership, partial COW and cache counters into `PrefixCacheManager`.
- Added host-only `IKvPageAllocator`; `PhysicalPagedKvCache` now implements the interface used by cache policy.
- Removed `PhysicalPagedKvCache::import_from_model`, the `CelegModel` friendship and the model/paged-KV header cycle.
- Added grouped concurrent metric domains while retaining the flat compatibility snapshot and C ABI v6.
- Split CMake into `celeg_host` and CUDA-dependent `celeg_core` targets.
- Removed the obsolete linear `common_prefix_length` policy helper.
- Added execution-plan, prefix-cache/fake-allocator and grouped-metrics tests.
- Added `SOLID.md` and an automated architecture boundary check.

## v0.0.13

- Added wavefront ragged packed prefill across unrelated active requests.
- Prompt rows now share GEMMs with `M = active prefill batch` while retaining independent positions, page tables, seen-token state and ShortConv state.
- Added per-row seen-token CUDA marking used by explicit-token packed prefill.
- Skip final normalization and the 65,536-row LM head on prefill waves where no request finishes its prompt.
- Added packed-prefill metrics, fallback/lane counters and maximum prefill batch reporting.
- Replaced linear longest-prefix cache scans with a token-ID radix tree.
- Added transactional radix insertion/removal synchronized with LRU cache ownership.
- Added partial-page COW that copies only initialized token intervals for BF16 and INT8 KV, including scale planes.
- Added C API v6 structures `celeg_packed_metrics_v2` and `celeg_paged_kv_metrics_v3`.
- Added host radix tests, CUDA source tests for per-row seen histories and partial-page cloning, host-only shared-library linkage and C ABI symbol validation.

## v0.0.12

- Added direct prompt prefill into the shared physical BF16/INT8 KV page arena.
- Reused persistent page-table and prompt-token staging buffers across prefill chunks.
- Concurrent paged lanes no longer allocate even a transient contiguous KV cache.
- Added continuation from restored prefix state so only the uncached prompt suffix is evaluated.
- Changed cache admission from exact-only lookup to longest compatible cached-prefix selection.
- Removed page-boundary restrictions from prefix caching.
- Added full-page copy-on-write for partial final pages both when inserting a cache entry and when reusing it.
- Added physical page cloning for BF16 and INT8 K/V plus INT8 scale planes.
- Added segmented paged GQA partial/reduction kernels for BF16 and INT8 long-context decode.
- Added automatic single-block versus segmented paged-attention selection.
- Hardened page retain/release transactions for duplicates, invalid IDs and partial-failure safety.
- Added C API v5 paged metrics for partial hits, reused tokens, COW pages, direct paged prefill and segmented paged decode.
- Added host tests for longest-prefix calculation and transactional page references, plus CUDA test coverage for page cloning and segmented paged attention.

## v0.0.11

- Replaced the concurrent engine's logical-only page accounting with a physical, shared GPU KV page arena.
- Added BF16 and INT8 paged KV layouts covering all six attention layers.
- Added direct paged K/V stores and strict/online paged GQA decode kernels.
- Routed packed continuous decode through per-request physical page tables.
- Added reference-counted exact-prefix reuse for page-aligned prompts.
- Added LRU prefix eviction by entry limit and under physical-page memory pressure, plus cache hit/miss/insert/eviction metrics.
- Restored logits, seen-token and ShortConv state on a prefix hit while preserving the receiving request's RNG seed.
- Released temporary contiguous lane KV after prefill import.
- Added C API v4 paged-KV metrics, a dedicated prefix-cache benchmark and GPU tests for paged BF16/INT8 kernels.
- Hardened cancellation, overflow, invalid lane/session operations and reference-counted page allocation.

## v0.0.10

### Packed continuous decode

- Added `PackedDecodeExecutor` and packed scheduler integration.
- Active compatible requests now share one 14-layer model pass.
- BF16 projections use GEMMs with `M = active_batch`.
- W8A16/W4A16 custom linear kernels accept the same packed row layout.
- Added batched sampling, embedding, QKV split, SwiGLU and state scattering.
- Added per-row RoPE positions and pointer-table BF16/INT8 KV operations.
- Added pointer-table strict/online GQA and ShortConv recurrent-state kernels.
- Added automatic fallback to the v9 lane backend for segmented attention,
  singleton batches or incompatible request state.

### Metrics and API

- Bumped the advertised C ABI version to 3 while preserving v1/v2 functions.
- Added `celeg_packed_metrics_v1` and
  `celeg_engine_get_packed_metrics`.
- Extended the concurrent benchmark with packed/lane counts, maximum batch and
  packed-path throughput.

### Validation

- Added CUDA tests for packed QKV split, row-interleaved SwiGLU and independent
  packed sampling rows.
- Generated PTX/CUBIN for 51 kernels on sm_80, sm_89, sm_90 and sm_120 without
  spills.

### Scope boundary

- Multi-request prefill is not yet ragged-packed.
- Physical KV remains contiguous per request; page-table GQA is deferred.
- Segmented long-context requests currently use lane fallback.

## v0.0.9

- Added shared-weight concurrent request lanes, continuous admission, priority,
  token budgeting, chunked prefill, async split decode and logical page policy.
- Added concurrent C API v2, TTFT/ITL metrics, example and benchmark.

## v0.0.8

- Added INT8 KV cache, persistent sessions, C API v1 and runtime metrics.
