# Celeg SOLID and Packed CUDA Execution Refactoring Plan

**Project:** `celsowm/celeg`  
**Primary branch:** `master`  
**Reviewed head:** `e2c88d8e92fcd468f1b9c818f4bde6b950f880d1`  
**Previous plan baseline:** `8907e01cadede5c1c28426d2362e9effb9a53a53`  
**Review date:** 2026-08-03  
**Scope:** runtime composition, dependency boundaries, resolved-model contracts, model-family extensibility, C/C++ API composition, serving, chat/tool calling, CUDA packed decode, ragged prefill, GEMM dispatch, and performance governance.

> This is a source-level review. The repository contains useful correctness and benchmark infrastructure, but this revision did not execute builds, tests, sanitizers, CUDA benchmarks, Nsight Systems, or Nsight Compute. Every runtime-sensitive phase below still requires hardware validation.

---

## 1. Status Legend

| Status | Meaning |
|---|---|
| **DONE** | The intended boundary or capability exists and the reviewed source supports the claim. |
| **PARTIAL** | Meaningful work exists, but the original acceptance criteria are not yet met. |
| **NOT STARTED** | The current source still substantially matches the problem statement. |
| **BLOCKED** | Work should not begin until a prerequisite correctness or characterization task is complete. |
| **REPLACED** | The old task no longer matches the current project direction and has been rewritten. |

---

## 2. Executive Summary

Celeg has improved materially since the original plan was written. The project now has stronger neutral checkpoint contracts, explicit architecture and checkpoint catalogs, CPU and CUDA model compilers, richer chat/tool-call support, additional model families, benchmark manifests, native-GGUF performance tooling, and a growing set of architecture boundaries.

However, the highest-risk findings from the original review remain present in the current source, especially in packed CUDA execution.

### 2.1 What has changed since the previous plan

The following work is already present and must no longer be described as entirely missing:

- neutral tensor and weight-repository contracts were extracted into `checkpoint/tensor.hpp` and `checkpoint/weight_repository.hpp`;
- `CheckpointView::gguf` and `ResolvedModel::is_gguf` were removed;
- generic tensor roles were separated from family-owned naming-policy implementations;
- CPU and CUDA model compiler concepts exist;
- architecture and checkpoint catalogs support `add`, duplicate detection, and `freeze`;
- baseline manifests, numerical-comparison helpers, compile measurement, and benchmark documentation exist;
- native same-file GGUF benchmarking against llama.cpp exists;
- tool-call codecs and conversation/generation support are split by family more than before;
- MiniCPM5 and SmolLM3 were added;
- native-GGUF fan-out reuse was added to the CUDA GEMM path;
- the project documentation direction changed: several architecture documents were deliberately removed and replaced with focused `API.md` and `BENCHMARK.md` material.

### 2.2 Critical findings that remain current

1. **`packed_execution.cu` is still a monolith.**  
   `PackedDecodeExecutorImpl` owns capacities, stream and cuBLAS resources, the GEMM dispatcher, plan lifetime, all GPU and pinned buffers, validation, metadata staging, attention, convolution, dense FFN, MoE, decode, prefill, metrics, and host session mutation.

2. **Packed attention still assumes output projection input width equals `hidden`.**  
   `run_attention_layer` calls the attention output projection with `k_width = shape_.hidden`, although the actual input width is the layer's `AttentionSpec::query_width()`.

3. **Packed dense FFN still uses a global intermediate width.**  
   Workspace sizing and execution use `shape_.intermediate`, even though `RuntimeTopology` already exposes `feed_forward_intermediates` and `max_feed_forward_intermediate`.

4. **Packed execution plans are compiled per call.**  
   Both `decode` and `prefill` call `CudaExecutionPlan::compile(...)` and clear `active_plan_` afterward.

5. **Packed compatibility checks are incomplete.**  
   `options_compatible` checks shared weights, context, weight mode, KV mode, fast attention, fused projections, and fused residuals, but omits other execution-relevant options such as GEMM backend, Lt workspace and heuristic settings, autotune, attention mode and thresholds, chunk sizing, offload/residency policy, and any compiled-program identity.

6. **Packed lifecycle validation contains an unreachable branch.**  
   `eligible` rejects every phase other than `Ready` before separately checking `DecodePending`.

7. **Ragged prefill allocates on every call.**  
   It creates a temporary `std::vector<uint8_t*> flat_seen` and a temporary `DeviceBuffer<uint8_t*> d_flat_seen`.

8. **Ragged prefill projects vocabulary logits for all flattened tokens.**  
   When any request finalizes, final norm and LM-head projection run on every flattened row, followed by selected-row scattering. The expensive work should run only on finalized rows.

9. **Ragged prefill timing is enqueue timing, not GPU execution timing.**  
   `std::chrono` surrounds asynchronous CUDA launches without a completion event or stream synchronization before recording the elapsed value. Host session state is also advanced before an explicit completion boundary is established.

10. **Persistent packed metadata invalidation is too weak.**  
    `session_layout_changed` compares only session owner pointers. Replaced cache storage, changed layer backing, or a storage-generation change may not trigger rebinding.

11. **The CPU backend still knows concrete checkpoint formats.**  
    `src/backend/cpu/weights.cpp` includes GGUF and SafeTensors repository headers, uses `dynamic_cast<const GgufRepository*>`, and carries a raw architecture naming-policy pointer into weight loading.

12. **Runtime composition remains hardcoded.**  
    `load_model_bootstrap` uses process-static built-in checkpoint and architecture catalogs. `create_builtin_architecture_catalog` centrally edits the list of every built-in family.

13. **`ResolvedModel` still overlaps responsibilities.**  
    It carries `ModelDefinition`, `ModelGraph`, `RuntimeTopology`, `WeightPlan`, capabilities, architecture identity, checkpoint profile, chat profile, diagnostic identity, and a raw non-owning naming-policy pointer.

14. **Architecture-family implementations remain broad.**  
    Family `architecture.cpp` files still perform probing, metadata decoding, topology construction, graph construction, weight-plan construction, profile assignment, capability assignment, and identity construction.

15. **The C API remains a large composition and mapping unit.**  
    `src/api/api.cpp` still owns handles, error translation, ABI validation, enum mapping, CPU/CUDA option mapping, backend construction, engine/model/tokenizer functions, and lifecycle entry points.

16. **Chat ownership improved, but capability representation can still be inconsistent.**  
    `ChatProfileCatalog` owns templates and codecs, while `ChatCapabilities` exposes a raw codec pointer and parallel boolean flags. The valid-state rules are not encoded in the type.

17. **The latest native fan-out optimization adds a manual scope protocol.**  
    `GemmDispatcher::begin_native_fanout` and `end_native_fanout` rely on callers pairing operations correctly before the source activation is overwritten. This should become an RAII scope or an immutable compiled fan-out operation.

18. **The `GemmDispatcher` documentation still mislabels switch extension as OCP.**  
    Extending a central dispatch switch may be a valid closed-domain design, but it is not Open/Closed compliance. The code and documentation should state whether the domain is intentionally closed.

### 2.3 Updated SOLID assessment

This score is a static architectural assessment, not a runtime-quality score.

| Principle | Current score | Current reason |
|---|---:|---|
| Single Responsibility | 4.5/10 | Stronger compiler/catalog concepts exist, but packed execution, CPU weight setup, C API, serving composition, architecture resolution, and GEMM dispatch remain broad. |
| Open/Closed | 5.8/10 | Catalogs and family modules help, but built-in registration, runtime bootstrap, chat profiles, backend selection, graph variants, and layout dispatch still require central edits. |
| Liskov Substitution | 7.0/10 | Capability interfaces are generally sound, but narrower backend constraints, chat-role support, raw borrowed state, and async completion semantics need stronger contracts. |
| Interface Segregation | 7.8/10 | Repository capabilities and serving views are relatively narrow; internal context bags and multi-role concrete services still weaken the result. |
| Dependency Inversion | 6.4/10 | Neutral repository contracts are a real improvement, but CPU checkpoint casts, naming-policy leakage, static built-in catalogs, and composition roots still invert dependencies incompletely. |
| **Overall** | **~6.3/10** | The architectural direction is good, but the most performance-sensitive implementation remains insufficiently separated and contains confirmed correctness/performance risks. |

### 2.4 Target outcome

The target remains approximately:

- **SRP:** 8.0+
- **OCP:** 8.0+
- **LSP:** 8.0+
- **ISP:** 8.5+
- **DIP:** 8.5+
- **Overall:** 8.3–8.7

The target does **not** require virtual dispatch in CUDA layer loops. The preferred pattern remains:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

---

## 3. Non-Negotiable Refactoring Rules

### 3.1 Preserve hot-path behavior

Do not introduce deep virtual object graphs into:

- transformer layer loops;
- attention launch loops;
- GEMM dispatch;
- MoE expert execution;
- decode scheduling;
- token sampling.

Use virtual interfaces or registries during composition and compilation. Execute compact immutable bindings, function pointers, tagged records, or direct concrete calls afterward.

### 3.2 Correctness before extraction

Do not split `packed_execution.cu` until the current behavior is characterized and the confirmed shape, allocation, timing, and final-row issues are fixed or covered by failing tests.

### 3.3 Separate policy from mechanism

Examples:

- checkpoint selection is policy; tensor reading is mechanism;
- architecture selection is policy; graph compilation is mechanism;
- attention-mode selection is policy; kernel launch is mechanism;
- weight-layout selection is policy; linear execution is mechanism;
- fan-out detection is policy; activation quantization reuse is mechanism.

### 3.4 Use capabilities for optional behavior

Keep optional repository operations in separate interfaces. Apply the same rule to:

- vision encoders;
- tokenizer providers;
- chat/tool support;
- backend limits;
- native checkpoint storage;
- fan-out-capable linear execution;
- asynchronous completion.

### 3.5 Avoid file-only refactoring

A valid SRP refactor changes ownership, lifetime, contracts, tests, and dependency direction. Merely moving methods into more files is not sufficient.

### 3.6 Explicitly classify closed domains

A switch is acceptable for a deliberately closed domain. Mark and document it as closed. Do not describe “add another switch case” as OCP compliance.

Suggested annotation:

```cpp
// CELEG_CLOSED_DOMAIN: LinearStorageKind
```

---

## 4. Target Architecture

```text
RuntimeBuilder
    -> immutable RuntimeContext
        ├── ArchitectureCatalog
        ├── CheckpointFormatCatalog
        ├── ChatProfileCatalog
        ├── TokenizerProviderCatalog
        ├── BackendFactoryCatalog
        ├── VisionProviderCatalog
        └── DiagnosticsSink

Checkpoint source
    -> ICheckpointFormat
        -> CheckpointMetadata
        -> IWeightRepository capabilities

CheckpointMetadata + architecture resolver
    -> ResolvedModel
        ├── authoritative neutral graph
        ├── resolved source tensor requests
        ├── model capabilities
        ├── token/numerical policies
        └── diagnostic provenance

ResolvedModel + backend options
    -> backend compiler
        -> immutable compiled program
        -> immutable execution plan
        -> compiled linear/operator bindings
        -> resource/workspace requirements

Compiled backend
    -> sessions
    -> packed/non-packed executors
    -> scheduler adapters
    -> serving/protocol adapters
```

A backend must not need architecture-specific tensor-name generation or concrete checkpoint repository types.

---

## 5. Updated Phase Overview

| Phase | Priority | Status | Goal |
|---|---:|---|---|
| 0 | P0 | **PARTIAL** | Finish correctness, allocation, timing, and performance baselines. |
| 1 | P0 | **NOT STARTED** | Fix confirmed packed shape, lifecycle, timing, allocation, and final-row defects. |
| 2 | P0 | **NOT STARTED** | Compile immutable CUDA plans and complete packed compatibility identity. |
| 3 | P0/P1 | **PARTIAL** | Finish checkpoint/backend dependency inversion and resolve weight names before compilation. |
| 4 | P1 | **NOT STARTED** | Introduce injectable `RuntimeContext` and `RuntimeBuilder`. |
| 5 | P1 | **NOT STARTED** | Make resolved-model ownership authoritative and remove duplicated execution state. |
| 6 | P1 | **PARTIAL** | Split family resolution by responsibility and register families through composition. |
| 7 | P1 | **BLOCKED** | Extract packed workspace, validator, metadata staging, operators, and pipelines after Phase 1. |
| 8 | P1 | **PARTIAL** | Refactor GEMM/layout dispatch, including safe native fan-out scope. |
| 9 | P1 | **NOT STARTED** | Split C API mapping, handles, factories, and entry-point modules. |
| 10 | P1/P2 | **PARTIAL** | Reduce serving duplication and make modalities/backend limits capability-driven. |
| 11 | P2 | **PARTIAL** | Make chat, codecs, roles, and tokenizer providers coherent and injectable. |
| 12 | P2 | **DECISION PENDING** | Declare graph operator families closed or add a compile-time registry model. |
| 13 | P2 | **PARTIAL** | Strengthen LSP, lifetime, completion, and backend-limit contracts. |
| 14 | P2 | **PARTIAL** | Expand automated boundary checks and stale-plan detection. |
| 15 | P0–P2 | **PARTIAL** | Continuously validate correctness, throughput, latency, memory, and compile cost. |
| 16 | P2/P3 | **REPLACED** | Align documentation with the current focused-doc strategy; do not recreate deleted docs blindly. |

---

# Part I — Safety and Confirmed Packed Corrections

## 6. Phase 0 — Finish the Safety Net

### Current status: PARTIAL

Already present:

- benchmark manifests and reproducibility tooling;
- numerical comparison helpers;
- compile-time measurement tooling;
- native same-file GGUF benchmark support;
- benchmark documentation;
- CPU/CUDA test coverage reported by prior commits;
- model-family manifests including MiniCPM5 and SmolLM3.

Still required:

- packed vs non-packed decode parity across supported paths;
- standard vs ragged prefill parity;
- synthetic `query_width != hidden` fixture;
- synthetic per-layer FFN-width fixture;
- allocation counters for host and device buffers;
- CUDA-event timing for packed decode and prefill;
- mixed `finalize` ragged tests;
- persistent-metadata rebinding tests;
- native-fan-out misuse tests;
- machine-readable before/after results for every packed milestone.

### Build/test matrix

At minimum:

| Dimension | Required values |
|---|---|
| Backend | CPU, CUDA |
| Build | Debug, RelWithDebInfo, Release |
| Checkpoint | GGUF, SafeTensors where supported, fake in-memory repository |
| Model family | LFM2/LFM2.5, Granite, Gemma4, MiniCPM5, SmolLM3 |
| Weight mode | BF16, INT8, INT4, native GGUF layouts |
| KV mode | BF16, INT8 |
| Execution | single, packed decode, standard prefill, ragged prefill |
| Mixer | attention, short convolution, hybrid |
| FFN | dense, MoE, per-layer dense widths |
| Attention | local, paged, segmented, shared KV, non-equal projection width |

### Acceptance criteria

- A failing regression test exists for every confirmed packed issue before the behavior is changed.
- CUDA metrics distinguish host preparation, GPU execution, host commit, and end-to-end time.
- Steady-state allocation count is measurable.
- Baseline results identify exact model file, hash, build commit, GPU, driver, CUDA version, options, warmup, and repetitions.

---

## 7. Phase 1 — Fix Packed Execution Before Modularizing It

### Current status: NOT STARTED

### 7.1 Use layer-specific attention widths

Current defect:

```cpp
linear(op_output.data(), *attention.out, hidden.data(), rows,
       shape_.hidden, shape_.hidden, beta);
```

The input width must be the layer's actual query projection width:

```cpp
const int query_width = attention.layout.query_width();
linear(op_output.data(), *attention.out, hidden.data(), rows,
       shape_.hidden, query_width, beta);
```

Compile and validate this value once rather than reading architecture identity in execution.

### 7.2 Use layer-specific dense FFN widths

Current execution and workspace use `shape_.intermediate`. Replace execution-time uses with the current layer's width and size workspace from the maximum.

Required invariant:

```text
workspace capacity = topology.max_feed_forward_intermediate
layer execution width = topology.feed_forward_intermediates[layer]
```

The compiled dense binding should include:

```cpp
struct PackedDenseFfnBinding {
    int intermediate = 0;
    const LinearWeight* w13 = nullptr;
    const LinearWeight* w2 = nullptr;
};
```

### 7.3 Centralize operation eligibility

Replace contradictory checks with one pure validator:

```cpp
enum class PackedOperation { Decode, Prefill };

PackedEligibility validate_session(
    const PackedSessionContext& session,
    PackedOperation operation,
    const PackedExecutorCapabilities& capabilities);
```

### 7.4 Remove per-prefill allocations

Move the following into persistent executor workspace:

- flattened seen-token pointer host storage;
- flattened seen-token pointer device storage;
- gathered hidden rows;
- gathered normalized rows;
- finalized-request indices;
- any temporary page-table expansion storage.

No `DeviceBuffer` construction or capacity-growing `std::vector` is allowed in steady-state prefill.

### 7.5 Gather finalized rows before final norm and LM head

Current behavior runs final norm and LM-head over all flattened tokens if any request finalizes.

Target sequence:

```text
transform all flattened tokens
build finalized-row indices
gather finalized hidden rows
final norm only finalized rows
LM head only finalized rows
scale/softcap only finalized rows
scatter one logits row per finalized request
```

Rows with `finalize == 0` must advance position and remain `Prefilling` without vocabulary projection.

### 7.6 Define completion semantics and fix metrics

Introduce CUDA events or an explicit completion object:

```cpp
struct PackedCompletion {
    CudaEvent completed;
    PackedTiming timing;
};
```

At minimum:

```cpp
struct PackedTiming {
    double host_prepare_ms = 0.0;
    double gpu_execute_ms = 0.0;
    double host_commit_ms = 0.0;
    double end_to_end_ms = 0.0;
};
```

Do not update host phase/position as completed until the operation's completion contract is satisfied.

### 7.7 Strengthen persistent metadata invalidation

Add a stable storage generation or binding fingerprint to each session context. Rebind when any relevant pointer table changes, not only when `owner` changes.

### Acceptance criteria

- synthetic non-equal query-width packed parity passes;
- per-layer FFN-width packed parity passes;
- no unreachable lifecycle branch remains;
- zero steady-state host/device allocation occurs in decode and prefill;
- LM-head rows equal finalized request count;
- partial prefill performs no logits projection;
- GPU timing correlates with profiler timing;
- session state is not partially committed after a failed launch or completion error;
- metadata rebinds after cache-storage replacement.

---

## 8. Phase 2 — Immutable CUDA Plans and Complete Compatibility

### Current status: NOT STARTED

Both packed paths currently compile `CudaExecutionPlan` per call. Compile it when model/session options become immutable.

### Target objects

```cpp
struct CudaPlanKey {
    WeightMode weight_mode;
    KvCacheMode kv_cache_mode;
    GemmBackend gemm_backend;
    AttentionMode attention_mode;
    int attention_chunk_tokens;
    int attention_auto_threshold;
    bool fast_attention;
    bool fused_projections;
    bool fused_residuals;
    size_t lt_workspace_bytes;
    int lt_heuristics;
    bool lt_autotune;
    uint64_t moe_policy_fingerprint;
    int max_context;
};

class CudaCompiledExecutionPlan {
public:
    uint64_t fingerprint() const noexcept;
};
```

The actual key must include every option that changes kernels, numerics, workspace, cache layout, offload/residency, fusion, or dispatch.

### Packed compatibility key

```cpp
struct PackedCompatibilityKey {
    const SharedModelWeights* weights = nullptr;
    uint64_t compiled_program_id = 0;
    uint64_t execution_plan_fingerprint = 0;
    int max_context = 0;
};
```

### Acceptance criteria

- no plan compile occurs in steady-state decode or prefill;
- plan compilation count is visible in diagnostics;
- batch validation compares immutable fingerprints;
- incompatible GEMM, attention, fusion, KV, workspace, autotune, and MoE policies are rejected;
- plans are immutable and thread-safe.

---

# Part II — Dependency Inversion and Neutral Contracts

## 9. Phase 3 — Finish the Checkpoint/Backend Boundary

### Current status: PARTIAL

Completed foundation:

- neutral `TensorDType`, `TensorLocator`, `HostTensorView`, and repository interfaces;
- separate optional location and random-access capabilities;
- removal of direct `CheckpointView::gguf` and `ResolvedModel::is_gguf` flags;
- family-owned naming-policy implementations.

Remaining violations:

- CPU backend includes concrete GGUF and SafeTensors headers;
- CPU backend detects native checkpoint storage with `dynamic_cast<const GgufRepository*>`;
- CPU backend receives and calls an architecture-owned naming policy;
- `ResolvedModel` carries an unresolved raw naming-policy pointer;
- native-source behavior is represented as concrete type identity rather than a neutral capability or resolved storage policy.

### 9.1 Add a neutral native-storage capability

Do not replace one concrete cast with another format flag. Model the actual need.

Possible contract:

```cpp
class INativeBlockTensorRepository {
public:
    virtual ~INativeBlockTensorRepository() = default;
    virtual std::optional<NativeBlockTensorView> native_block_view(
        std::string_view resolved_name) const = 0;
};
```

Or represent the choice in resolved weight requests:

```cpp
struct ResolvedWeightRequest {
    WeightSemantic semantic;
    std::string source_tensor_name;
    TensorShape source_shape;
    TensorShape destination_shape;
    WeightTransform transform;
    NativeStoragePreference native_storage;
    int layer = -1;
};
```

### 9.2 Resolve names before backend compilation

The architecture layer should turn logical tensor roles into concrete source candidates and select the existing source name. Backend code should not call `ITensorNamingPolicy`.

### 9.3 Separate pack-cache policy from format identity

CPU pack-cache decisions should depend on resolved storage/layout capabilities and requested output format, not `GgufRepository` identity.

### Acceptance criteria

- backend directories do not include concrete GGUF/SafeTensors repository headers;
- no backend casts to a concrete checkpoint repository;
- no backend receives `ITensorNamingPolicy`;
- all source names and expected shapes are resolved before backend compilation;
- fake in-memory repositories load through CPU and CUDA compiler tests;
- native GGUF paths continue to preserve performance and memory behavior.

---

## 10. Phase 4 — Injectable Runtime Composition

### Current status: NOT STARTED

`ArchitectureCatalog` and `CheckpointFormatCatalog` already support registration and freezing, but bootstrap creates process-static built-in catalogs and built-ins are centrally enumerated.

### Target API

```cpp
class RuntimeContext {
public:
    const ArchitectureCatalog& architectures() const;
    const CheckpointFormatCatalog& checkpoint_formats() const;
    const ChatProfileCatalog& chat_profiles() const;
    const TokenizerProviderCatalog& tokenizer_providers() const;
    const BackendFactoryCatalog& backends() const;
    const VisionProviderCatalog& vision_providers() const;
};

class RuntimeBuilder {
public:
    RuntimeBuilder& add_builtins();
    RuntimeBuilder& add_architecture(std::unique_ptr<IArchitecture>);
    RuntimeBuilder& add_checkpoint_format(std::unique_ptr<ICheckpointFormat>);
    RuntimeBuilder& add_chat_profile(ChatProfileRegistration);
    RuntimeBuilder& add_tokenizer_provider(std::unique_ptr<ITokenizerProvider>);
    RuntimeBuilder& add_backend_factory(std::unique_ptr<IBackendFactory>);
    RuntimeBuilder& add_vision_provider(std::unique_ptr<IVisualEmbeddingProviderFactory>);
    RuntimeContext build();
};
```

### Migration

```cpp
ModelBootstrap load_model_bootstrap(
    const RuntimeContext& runtime,
    const std::filesystem::path& model_path);
```

A convenience default may delegate to a process-immutable built-in context, but internal code and tests must be able to inject a context.

### Family registration

```cpp
void register_lfm2(RuntimeBuilder&);
void register_granite(RuntimeBuilder&);
void register_gemma4(RuntimeBuilder&);
void register_minicpm5(RuntimeBuilder&);
void register_smollm3(RuntimeBuilder&);
```

### Acceptance criteria

- custom architecture and checkpoint-format tests require no Celeg source edit;
- duplicate IDs fail explicitly;
- catalog lifetime and thread safety are documented;
- built-in registration is isolated in one composition module;
- no mutable global catalog exists;
- bootstrap, C API, CLI, and serving can receive a runtime context.

---

## 11. Phase 5 — Make `ResolvedModel` Authoritative

### Current status: NOT STARTED

Current overlapping data includes:

- `ModelDefinition`;
- `ModelGraph`;
- `RuntimeTopology`;
- `WeightPlan`;
- capabilities;
- architecture/checkpoint/chat IDs;
- profile objects;
- raw naming-policy pointer;
- diagnostic identity.

### Target decomposition

```cpp
struct ModelDimensions {
    int hidden;
    int layers;
    int vocab_size;
    int maximum_attention_projection_width;
    int maximum_attention_head_dim;
    int maximum_dense_intermediate;
    int maximum_moe_intermediate;
};

struct TokenPolicy {
    std::optional<int32_t> bos;
    std::vector<int32_t> eos;
    std::optional<int32_t> pad;
};

struct NumericalPolicy {
    float norm_eps;
    float embedding_multiplier;
    float residual_multiplier;
    float logits_divisor;
    float final_logit_softcap;
};

struct ModelProvenance {
    std::string architecture_id;
    std::string checkpoint_format_id;
    std::string checkpoint_profile_id;
    std::string source_description;
};

struct ResolvedModel {
    ModelDimensions dimensions;
    TokenPolicy tokens;
    NumericalPolicy numerics;
    ModelGraph graph;
    ResolvedWeightPlan weights;
    ModelCapabilities capabilities;
    TextProfileReference text_profile;
    ModelProvenance provenance;
};
```

### Rules

- one authoritative owner per value;
- per-layer dimensions are explicit in the graph/program;
- maxima are derived from per-layer data;
- compiled programs contain execution state only;
- provenance never drives backend execution;
- raw policy pointers are removed.

### Acceptance criteria

- no duplicated hidden/layer/head/token/numerical fields remain without a documented derivation;
- backend execution does not branch on architecture or checkpoint identity;
- workspace requirements derive from the compiled program;
- all lifetimes are value-owned, shared immutable, or referenced through stable IDs.

---

# Part III — Model Families and Packed Subsystems

## 12. Phase 6 — Split Architecture-Family Responsibilities

### Current status: PARTIAL

Progress:

- each major family has its own source directory;
- naming policies are family-owned;
- tool-call codecs are increasingly family-owned;
- generic dense graph and weight-plan builders exist.

Remaining issue:

A family `architecture.cpp` still commonly performs probing, metadata access, validation, topology building, graph building, weight-plan building, identity/profile assignment, token extraction, and capability assignment.

### Target family layout

```text
src/models/<family>/
  registration.cpp
  probe.cpp
  metadata.cpp
  validation.cpp
  topology.cpp
  graph.cpp
  weights.cpp
  capabilities.cpp
  naming_policy.cpp
  text_profile.cpp
  tool_call_codec.cpp
  architecture.cpp
```

Not every family needs every file. The boundaries matter more than the file count.

### Acceptance criteria

- metadata decoding tests need no backend construction;
- graph tests need no checkpoint I/O;
- resolved-weight tests validate concrete names and shapes independently;
- adding a family edits only its registration and composition entry;
- backend code contains no family identity switch.

---

## 13. Phase 7 — Refactor Packed Execution into Owned Subsystems

### Current status: BLOCKED by Phase 1

Do not begin this extraction until Phase 1 tests and fixes are in place.

### Target modules

```text
src/backend/cuda/model/packed/
  compatibility.hpp/.cpp
  validator.hpp/.cpp
  workspace.hpp/.cu
  metadata_stager.hpp/.cu
  compiled_program.hpp/.cpp
  embedding_executor.hpp/.cu
  attention_executor.hpp/.cu
  convolution_executor.hpp/.cu
  dense_ffn_executor.hpp/.cu
  moe_executor.hpp/.cu
  transformer_executor.hpp/.cu
  final_projection.hpp/.cu
  decode_pipeline.hpp/.cu
  prefill_pipeline.hpp/.cu
  metrics.hpp/.cpp
  executor.cpp
```

### Required ownership boundaries

#### `PackedWorkspace`

Own every reusable host/device buffer and capacity. No pipeline owns scratch allocation.

#### `PackedBatchValidator`

Pure host-side validation, structured error codes, no CUDA dependency.

#### `PackedMetadataStager`

Own persistent pointer binding, step metadata, ragged flattening, page-table staging, and storage-generation tracking.

#### `PackedCompiledProgram`

Precompute layer kinds, dimensions, weights, KV owner/slot, fusion behavior, residual policy, and compiled linear bindings.

#### Operator executors

Attention, convolution, dense FFN, and MoE receive narrow explicit contexts and do not mutate session lifecycle.

#### Decode/prefill pipelines

Orchestration only. Completion and host commit are separate steps.

### Acceptance criteria

- `PackedDecodeExecutorImpl` no longer owns all packed responsibilities;
- the transformer loop does not repeatedly call `as_attention`, `as_convolution`, `as_dense_ffn`, or `as_moe_ffn`;
- dimensions are immutable and prevalidated;
- no architecture identity appears in execution;
- extraction causes no unexplained throughput, latency, or VRAM regression.

---

# Part IV — CUDA Linear Dispatch

## 14. Phase 8 — Refactor `GemmDispatcher` and Weight-Layout Binding

### Current status: PARTIAL

Current strengths:

- GEMM dispatch is extracted from the compiled model;
- cuBLAS/cuBLASLt and native GGUF paths are centralized;
- Lt plan caching/autotuning exists;
- native activation fan-out reuse exists.

Current problems:

- one object owns handles, Lt cache, autotuning, Lt workspace, MMQ workspace, BF16/INT8/INT4/native layouts, GEMV fallback, storage dispatch, and fan-out scope state;
- comments describe central switch extension as OCP;
- manual `begin_native_fanout`/`end_native_fanout` pairing is error-prone;
- packed execution bypasses a compile-time weight-layout binding and repeats storage-kind decisions in execution;
- dispatcher option lifetime and mutability contracts should be explicit.

### 14.1 Introduce an RAII fan-out scope

```cpp
class NativeFanoutScope {
public:
    NativeFanoutScope(GemmDispatcher&, const __nv_bfloat16* x, int m, int k);
    ~NativeFanoutScope();

    NativeFanoutScope(const NativeFanoutScope&) = delete;
    NativeFanoutScope& operator=(const NativeFanoutScope&) = delete;
};
```

Better long-term target: compile a projection group that quantizes once and executes all bound projections without mutable ambient scope state.

### 14.2 Separate collaborators

```text
CudaLinearRuntime
  ├── CublasHandles
  ├── LtPlanCache
  ├── LtAutotuner
  ├── LinearWorkspace
  ├── NativeQuantWorkspace
  └── compiled linear bindings
```

### 14.3 Compile linear bindings

```cpp
using LinearExecutionFn = void (*)(
    const LinearExecutionRequest&,
    CudaLinearRuntime&);

struct CompiledLinearBinding {
    LinearExecutionFn execute = nullptr;
    LinearKernelKind kind;
    const LinearWeight* weight = nullptr;
};
```

Virtual/registry selection occurs at compile time; execution uses the binding.

### Acceptance criteria

- fan-out scopes are exception-safe and cannot remain accidentally active;
- nested, mismatched-input, and overwritten-source cases are tested;
- plan cache and autotuner are independently testable;
- new layouts do not require edits to packed operator code;
- hot-path overhead does not regress measurably;
- comments correctly distinguish closed domains from extensible registries.

---

# Part V — API, Serving, Chat, and Graph Contracts

## 15. Phase 9 — Split the C API

### Current status: NOT STARTED

Target layout:

```text
src/api/
  handles.hpp
  errors.cpp
  validation.cpp
  generation_mapping.cpp
  cpu_option_mapping.cpp
  cuda_option_mapping.cpp
  backend_factory.cpp
  model_api.cpp
  engine_api.cpp
  tokenizer_api.cpp
  diagnostics_api.cpp
```

### Rules

- preserve opaque handles and versioned structs;
- no exception crosses the C boundary;
- add handle-kind tags and optional debug generations;
- backend construction goes through factories or runtime context;
- enum switches are treated as ABI mapping, not backend composition;
- mapping functions have table-driven tests.

### Acceptance criteria

- `src/api/api.cpp` is no longer a monolith;
- adding a backend does not edit unrelated model/tokenizer entry points;
- CPU and CUDA mappings are independently tested;
- ABI compatibility tests remain green.

---

## 16. Phase 10 — Refactor Serving Composition

### Current status: PARTIAL

The project already has service views and a `ServiceBundle`, but concrete services and composition roots still combine lifecycle, scheduler control, diagnostics, tokenizer/profile selection, modality selection, backend construction, and protocol behavior.

### Target roles

```cpp
class IRequestService;
class IRequestResultStore;
class ISchedulerController;
class IServiceDiagnostics;
```

One object may implement several roles, but callers should depend on the narrow role they need.

### Required changes

- extract backend-neutral request lifecycle;
- share CPU/CUDA orchestration where semantics match;
- publish backend limits explicitly;
- make vision selection capability-driven;
- remove fixed profile-string and fixed mmproj filename conventions from generic composition;
- obtain tokenizers through providers rather than concrete checkpoint casts;
- define cancellation and asynchronous completion semantics consistently.

### Acceptance criteria

- main composition roots are small and declarative;
- CPU/CUDA lifecycle duplication is reduced;
- no implementation silently narrows a public request contract;
- modalities and tokenizer selection are provider/capability-driven.

---

## 17. Phase 11 — Make Chat and Tokenizer Composition Coherent

### Current status: PARTIAL

Current progress:

- conversation and generation abstractions exist;
- family-specific tool-call codecs exist;
- the catalog owns templates and codecs;
- MiniCPM5 and SmolLM3 tool behavior is represented.

Remaining issues:

- built-in profile construction is central;
- template classes are declared centrally;
- `ChatCapabilities` mixes raw codec pointer and booleans;
- role support is represented incompletely;
- tokenizer/provider composition is not part of a runtime context.

### Target registration

```cpp
struct ChatRoleCapabilities {
    bool system;
    bool developer;
    bool user;
    bool assistant;
    bool tool;
};

struct ChatProfileRegistration {
    std::string id;
    std::shared_ptr<const IChatTemplate> template_engine;
    std::shared_ptr<const IChatToolCallCodec> tool_codec;
    ChatRoleCapabilities roles;
    ToolCapabilities tools;
    MultimodalPromptCapabilities multimodal;
};
```

Invalid combinations must fail during registration.

### Acceptance criteria

- capability state cannot say “native codec supported” while codec is null;
- unsupported roles fail through a structured result or pre-validation;
- profiles and tokenizer providers register through `RuntimeBuilder`;
- adding a profile does not edit one central template implementation file;
- serving does not cast checkpoint repositories to create tokenizers.

---

## 18. Phase 12 — Declare the Graph Extension Model

### Current status: DECISION PENDING

Closed variants such as attention/convolution and dense/MoE remain reasonable for a performance runtime if Celeg intentionally owns the operator universe.

### Option A — Closed operator universe

Keep variants and document that adding an operator requires coordinated graph, CPU compiler, CUDA compiler, tests, and diagnostics changes.

### Option B — Open compiler registry

Use stable operator IDs and compile-time registries, while execution still receives concrete backend bindings.

### Recommendation

Use a two-level design:

1. a small explicit neutral set for stable common categories;
2. backend-specific compiled bindings;
3. only introduce a public operator registry after at least one genuinely new family such as Mamba/DeltaNet/RWKV demonstrates the need.

### Acceptance criteria

- the repository explicitly states whether each operator domain is open or closed;
- unsupported backend/operator pairs fail during compilation;
- no architecture switch appears in backend execution loops;
- the change surface for adding one synthetic operator is measured.

---

## 19. Phase 13 — Strengthen Contracts and Lifetimes

### Current status: PARTIAL

Audit and formalize:

- packed session borrowed pointers;
- storage invalidation generations;
- runtime-context-owned catalogs;
- naming-policy lifetime until removed;
- chat codec ownership;
- `GemmDispatcher` option lifetime;
- native fan-out scope lifetime;
- service limits;
- asynchronous prefill/decode completion;
- cancellation and failure rollback.

### Acceptance criteria

- no undocumented narrower backend precondition exists;
- borrowed pointers document owner, mutation, invalidation, and thread safety;
- async operations expose a completion contract;
- raw policy pointers are removed or replaced by stable immutable ownership;
- CPU/CUDA contract tests run against the same request semantics.

---

# Part VI — Governance, Performance, and Documentation

## 20. Phase 14 — Expand Automated Boundaries

### Current status: PARTIAL

Keep existing checks and add rules that reject:

- concrete checkpoint repository headers/types under backend directories;
- architecture IDs in backend execution;
- chat-profile string branching outside registration/composition/diagnostics;
- `CudaExecutionPlan::compile` in decode/prefill/layer paths;
- `DeviceBuffer` construction or `cudaMalloc` in steady-state execution;
- unresolved `ITensorNamingPolicy` use in backends;
- manual fan-out scope use without an RAII guard;
- claims that extending a switch is OCP unless the domain is explicitly registry-driven.

Add extension tests for:

- custom architecture;
- custom checkpoint format;
- custom chat profile;
- custom tokenizer provider;
- fake backend factory if public;
- fake repository through CPU and CUDA compilation.

### Plan maintenance rule

This document must be updated whenever a milestone is completed. Every status change should include:

- commit SHA;
- tests added;
- benchmark artifact;
- remaining deviations.

### Acceptance criteria

- CI rejects dependency-direction regressions;
- hot-path plan compilation and obvious allocation regressions are caught;
- public extension tests edit no built-in central file;
- stale checked tasks are linked to evidence or reverted to partial.

---

## 21. Phase 15 — Continuous Performance Validation

### Current status: PARTIAL

The repository has meaningful benchmark tooling. Expand it around the refactoring milestones rather than creating a separate final optimization phase.

### Required packed measurements

#### Decode

- batch 1, 2, 4, 8, 16, maximum supported;
- aggregate tokens/s;
- per-request p50/p95/p99;
- kernels/token;
- H2D and D2H bytes/token;
- synchronization count;
- host prepare/GPU execute/host commit;
- device and host allocation count;
- VRAM peak and workspace size.

#### Ragged prefill

Use at least:

```text
[16, 32, 64, 128]
[4, 512, 32, 2048]
[1024, 1024, 1024, 1024]
mixed finalize flags
all non-finalizing rows
```

Measure:

- flattened tokens/s;
- finalized rows;
- actual LM-head rows;
- logits scratch size;
- page-table copy volume;
- allocation count;
- host prepare/GPU execute/host commit.

#### Shapes

- `query_width == hidden`;
- `query_width != hidden`;
- constant and per-layer FFN widths;
- dense and MoE;
- attention and convolution;
- shared KV;
- local, paged, and segmented attention;
- BF16, INT8, INT4, native Q4_K/Q6_K;
- native fan-out enabled and disabled.

### Initial regression thresholds

| Metric | Allowed regression |
|---|---:|
| Correctness | zero unexplained regressions |
| Decode throughput | <= 2% |
| Prefill throughput | <= 2% |
| p95 latency | <= 3% |
| VRAM | <= 2%, unless an explicit speed trade is documented |
| Host allocation | zero in steady-state packed execution |
| Device allocation | zero in steady-state packed execution |

### Acceptance criteria

- every CUDA structural change has a machine-readable comparison;
- event timing agrees with external profiling within expected tolerance;
- same-file/same-hash benchmark rules are preserved;
- fan-out reuse has both correctness and performance evidence.

---

## 22. Phase 16 — Documentation Aligned with Current Project Direction

### Current status: REPLACED

The previous plan required recreating a large documentation suite. That is stale because architecture documents were deliberately removed and focused API/benchmark guides were introduced.

Do not recreate deleted files automatically.

### Near-term documentation policy

Keep the following authoritative while contracts are changing:

- `README.md` — supported models, formats, quick start, project status;
- `API.md` — stable C API and ownership contracts;
- `BENCHMARK.md` — reproducible benchmark methodology;
- this plan — architecture debt, phase status, acceptance criteria;
- `AGENTS.md` — contributor/refactoring constraints;
- focused code-level diagnostics and generated API docs.

### Add only when the interfaces stabilize

- `docs/EXTENDING_CELEG.md` after `RuntimeBuilder` is real;
- `docs/PACKED_EXECUTION.md` after packed ownership and completion contracts stabilize;
- `docs/BACKEND_COMPILATION.md` after resolved weight and compiled binding contracts stabilize;
- migration notes when public interfaces actually change.

### Acceptance criteria

- documentation describes implemented APIs, not planned names as if they exist;
- deleted architecture docs are not recreated as stale snapshots;
- extension examples compile in CI;
- packed metrics, completion, compatibility, and workspace semantics are documented after implementation.

---

# Part VII — Recommended Commit Sequence

## 23. Immediate Milestone Order

### Milestone A — Confirmed packed correctness

1. Add non-equal attention-width fixture.
2. Add per-layer FFN-width fixture.
3. Fix attention output projection width.
4. Fix dense FFN layer width and workspace maximum.
5. Centralize lifecycle validation.
6. Add complete compatibility-key tests.

### Milestone B — Prefill completion and allocation

1. Add CUDA-event timing.
2. Define synchronous/asynchronous completion contract.
3. Move flattened seen-pointer buffers into workspace.
4. Add allocation assertions.
5. Gather finalized rows.
6. Run final norm and LM head only on finalized rows.
7. Test mixed and zero-finalize batches.

### Milestone C — Immutable plans

1. Define complete `CudaPlanKey`.
2. Compile plan during model/session resource creation.
3. Add plan fingerprint.
4. Remove per-call plan compilation.
5. Use fingerprint in packed compatibility.

### Milestone D — Checkpoint/backend boundary

1. Add neutral native-storage capability or resolved storage policy.
2. Resolve tensor source names before backend compilation.
3. Remove CPU concrete repository includes/casts.
4. Remove backend naming-policy dependency.
5. Add fake repository tests.
6. Strengthen boundary checker.

### Milestone E — Runtime composition

1. Implement immutable `RuntimeContext`.
2. Implement `RuntimeBuilder`.
3. Wrap built-ins in family registration functions.
4. Inject runtime into bootstrap.
5. Inject runtime into CLI, C API, and serving.
6. Add public extension tests.

### Milestone F — Packed subsystem extraction

1. Extract workspace requirements and ownership.
2. Extract validator and compatibility.
3. Extract metadata binding and staging.
4. Compile layer bindings.
5. Extract operator executors.
6. Split decode and prefill pipelines.
7. Compare performance after every extraction.

### Milestone G — GEMM and native fan-out safety

1. Add RAII fan-out guard.
2. Add fan-out lifecycle tests.
3. Separate Lt cache/autotuner/workspace.
4. Compile linear bindings.
5. Correct closed-domain/OCP documentation.

### Milestone H — Broader SRP cleanup

1. Make `ResolvedModel` authoritative.
2. Split family resolution stages.
3. Split C API.
4. Reduce serving duplication.
5. Complete chat/tokenizer providers.
6. Decide graph extension model.
7. Publish stabilized extension documentation.

---

# Part VIII — Actionable Backlog

## 24. P0 Backlog

### Packed shape correctness

- [ ] Add synthetic `query_width != hidden` fixture.
- [ ] Add synthetic per-layer FFN-width fixture.
- [ ] Use `attention.layout.query_width()` for output projection input.
- [ ] Use the current layer intermediate in dense FFN execution.
- [ ] Size dense workspace by `max_feed_forward_intermediate`.
- [ ] Add layer-indexed shape diagnostics.

### Packed compatibility and lifecycle

- [ ] Define complete `CudaPlanKey`.
- [ ] Define compiled-program ID.
- [ ] Define stable plan fingerprint.
- [ ] Replace partial `options_compatible`.
- [ ] Centralize decode/prefill eligibility.
- [ ] Remove unreachable `DecodePending` branch.
- [ ] Add storage-generation metadata invalidation.

### Prefill correctness and performance

- [ ] Move flat seen host/device storage into workspace.
- [ ] Add finalized-row index storage.
- [ ] Gather final hidden rows.
- [ ] Final-normalize only gathered rows.
- [ ] Project LM-head only gathered rows.
- [ ] Test mixed finalize.
- [ ] Test zero finalize.
- [ ] Test failure before host commit.

### Timing and allocation

- [ ] Add reusable CUDA event timer.
- [ ] Separate host prepare/GPU execute/host commit/end-to-end.
- [ ] Define completion semantics.
- [ ] Instrument `DeviceBuffer` allocation.
- [ ] Instrument relevant host allocations.
- [ ] Assert zero steady-state allocation.

### Immutable plans

- [ ] Compile CUDA plan outside packed call paths.
- [ ] Expose plan compile count.
- [ ] Remove `active_plan_` set/reset per call.
- [ ] Reject mixed plan fingerprints in a packed batch.

---

## 25. P1 Backlog

### Checkpoint/backend boundary

- [x] Extract neutral tensor contracts.
- [x] Extract optional repository capabilities.
- [x] Remove `CheckpointView::gguf` and `ResolvedModel::is_gguf`.
- [x] Move family naming policies out of generic roles.
- [ ] Model native block storage as a neutral capability/policy.
- [ ] Resolve concrete source names before backend compilation.
- [ ] Remove backend concrete repository includes/casts.
- [ ] Remove backend naming-policy pointer.
- [ ] Add fake repository CPU/CUDA tests.

### Runtime composition

- [x] Add/freeze architecture catalog.
- [x] Add/freeze checkpoint-format catalog.
- [ ] Implement `RuntimeContext`.
- [ ] Implement `RuntimeBuilder`.
- [ ] Register family bundles.
- [ ] Inject runtime into bootstrap.
- [ ] Add tokenizer/backend/vision provider catalogs.
- [ ] Add public custom-extension tests.

### Packed ownership

- [ ] Create `PackedWorkspaceRequirements`.
- [ ] Extract workspace.
- [ ] Extract pure validator.
- [ ] Extract metadata stager.
- [ ] Compile packed layer program.
- [ ] Extract attention executor.
- [ ] Extract convolution executor.
- [ ] Extract dense FFN executor.
- [ ] Extract MoE executor.
- [ ] Split decode and prefill pipelines.

### GEMM/layout dispatch

- [x] Extract `GemmDispatcher` from compiled model.
- [x] Add Lt cache/autotune infrastructure.
- [x] Add native GGUF linear paths.
- [x] Add native activation fan-out reuse.
- [ ] Replace manual fan-out scope with RAII/compiled group.
- [ ] Separate handles, cache, autotuner, and workspaces.
- [ ] Compile linear execution bindings.
- [ ] Correct OCP/closed-domain comments.

### API and serving

- [ ] Split C API modules.
- [ ] Add backend factory catalog.
- [ ] Extract shared request lifecycle.
- [ ] Publish backend limits.
- [ ] Make vision provider-driven.
- [ ] Remove tokenizer repository casts.

---

## 26. P2 Backlog

### Resolved model

- [ ] Remove duplicated definition/topology fields.
- [ ] Split provenance from execution.
- [ ] Define authoritative token policy.
- [ ] Define authoritative numerical policy.
- [ ] Make per-layer values authoritative.
- [ ] Remove raw policy pointers.

### Architecture families

- [x] Move families into dedicated directories.
- [x] Add MiniCPM5.
- [x] Add SmolLM3.
- [x] Split several naming policies and tool codecs.
- [ ] Split probe/metadata/topology/graph/weights/capabilities.
- [ ] Add family registration functions.
- [ ] Add stage-focused unit tests.

### Chat and tokenizer

- [x] Add conversation/generation abstractions.
- [x] Add family tool-call codecs.
- [x] Catalog-own templates/codecs.
- [ ] Replace raw codec pointer in capabilities.
- [ ] Encode role capabilities.
- [ ] Add tokenizer providers.
- [ ] Register through runtime context.

### Governance and docs

- [x] Add benchmark manifests and numerical helpers.
- [x] Add focused API and benchmark guides.
- [ ] Add packed allocation/plan boundary checks.
- [ ] Add public extension compilation tests.
- [ ] Add evidence links for plan status transitions.
- [ ] Create extension/packed docs only after interfaces stabilize.

---

# Part IX — Definition of Done

## 27. Architectural Definition of Done

The refactoring is complete when:

- runtime composition is injectable;
- built-in families register without central runtime switches;
- backends do not depend on concrete checkpoint formats;
- tensor names and transforms are resolved before backend compilation;
- resolved-model fields have one authoritative owner;
- compiled programs contain only execution-relevant state;
- optional behavior is capability-driven;
- raw borrowed policy lifetimes are removed or formally safe;
- graph domains are explicitly open or closed;
- C API and serving composition roots are small and declarative;
- boundary CI enforces the dependency direction.

## 28. Packed Execution Definition of Done

Packed execution is complete when:

- non-equal attention widths and variable FFN widths pass;
- no per-call CUDA plan compilation occurs;
- compatibility uses immutable complete fingerprints;
- workspace is a dedicated component;
- validation is pure host-side logic;
- metadata staging is isolated and generation-aware;
- layer dimensions and bindings are compiled once;
- attention, convolution, dense FFN, and MoE execution are separate;
- decode and prefill are separate orchestration pipelines;
- prefill projects logits only for finalized rows;
- zero steady-state host/device allocation occurs;
- metrics measure actual GPU work;
- completion and host commit semantics are explicit;
- no partial state remains after failure;
- performance stays within the agreed thresholds.

## 29. Final Priority

The highest-value order is now:

1. **Fix confirmed packed shape defects.**
2. **Fix prefill allocation, final-row projection, completion, and timing.**
3. **Compile immutable plans and complete compatibility identity.**
4. **Finish checkpoint/backend dependency inversion.**
5. **Introduce runtime composition.**
6. **Extract packed ownership boundaries.**
7. **Make native fan-out scope safe and compile linear bindings.**
8. **Make resolved-model ownership authoritative.**
9. **Split architecture, C API, serving, and chat responsibilities.**
10. **Stabilize and document extension contracts.**

The project should optimize for clear ownership and precompiled execution, not merely shorter source files.
