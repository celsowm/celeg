# Celeg SOLID and Packed CUDA Execution Refactoring Plan

**Project:** `celsowm/celeg`  
**Primary branch:** `master`  
**Reviewed code head:** `848957c364ed19b9ad2527ea1bc5a2f6e7d81d32`  
**Previous plan baseline:** `8907e01cadede5c1c28426d2362e9effb9a53a53`  
**Review date:** 2026-08-03  
**Scope:** runtime composition, dependency boundaries, resolved-model contracts, model-family extensibility, C/C++ API composition, serving, chat/tool calling, CUDA packed decode, ragged prefill, native GGUF/MMQ dispatch, and performance governance.

> **Concurrent-commit note:** commit `848957c364ed19b9ad2527ea1bc5a2f6e7d81d32` landed while this review was in progress, immediately before the first documentation update. The documentation commit was created as a direct descendant of it, so no source changes were overwritten. This revision corrects the reviewed head and incorporates its MMQ tensor-core behavior.

> This is a source-level review. This revision did not execute builds, tests, sanitizers, CUDA benchmarks, Nsight Systems, or Nsight Compute. Runtime-sensitive conclusions must be validated on the supported hardware matrix.

---

## 1. Status Legend

| Status | Meaning |
|---|---|
| **DONE** | The intended boundary or capability exists and the reviewed source supports the claim. |
| **PARTIAL** | Meaningful work exists, but the acceptance criteria are not yet met. |
| **NOT STARTED** | The reviewed source still substantially matches the problem statement. |
| **BLOCKED** | Work should not start before a prerequisite correctness or characterization task. |
| **DECISION PENDING** | The project must explicitly choose an extension model before implementation. |
| **REPLACED** | The old task no longer matches the current project direction. |

---

## 2. Executive Summary

Celeg has improved materially since the original plan. It now has neutral checkpoint contracts, explicit architecture and checkpoint catalogs, CPU and CUDA compiler concepts, model-family directories, richer chat/tool-call support, reproducible benchmark tooling, native-GGUF execution, and support for MiniCPM5 and SmolLM3.

The project is not architecturally broken. Its main problem is that several intended abstractions are more mature than the implementations using them. The largest correctness, performance, ownership, and maintainability concentration remains the packed CUDA path.

### 2.1 Work already present

The following items must not be treated as entirely missing:

- neutral `TensorDType`, `TensorLocator`, `HostTensorView`, and `IWeightRepository` contracts;
- separate optional repository capabilities for location and random-access reads;
- removal of `CheckpointView::gguf` and `ResolvedModel::is_gguf`;
- family-owned tensor naming-policy implementations;
- CPU and CUDA model compiler concepts;
- architecture and checkpoint catalogs with duplicate detection and freezing;
- benchmark manifests, numerical-comparison helpers, compile measurement, and benchmark guides;
- native same-file GGUF benchmarking against llama.cpp;
- model-family tool-call codecs and conversation/generation support;
- MiniCPM5 and SmolLM3 architecture support;
- native-GGUF activation fan-out reuse in CUDA linear execution;
- MMQ tensor-core prefill enabled by default on detected `sm_72+` devices;
- focused `API.md` and `BENCHMARK.md` documentation replacing several removed architecture snapshots.

### 2.2 Confirmed findings still current

1. **`packed_execution.cu` is still a monolith.**  
   `PackedDecodeExecutorImpl` owns capacities, CUDA resources, GEMM dispatch, plan lifetime, all host/device buffers, validation, metadata staging, attention, convolution, dense FFN, MoE, decode, prefill, metrics, and host-state mutation.

2. **Packed attention still assumes output projection input width equals `hidden`.**  
   `run_attention_layer` uses `shape_.hidden` as the output projection input width instead of the layer's `AttentionSpec::query_width()`.

3. **Packed dense FFN still uses a global intermediate width.**  
   Workspace and execution use `shape_.intermediate` although topology already carries per-layer FFN widths and a maximum.

4. **Packed execution plans are compiled per call.**  
   Both packed decode and packed prefill call `CudaExecutionPlan::compile(...)`, store the plan in mutable ambient state, then clear it.

5. **Packed compatibility checks are incomplete.**  
   Current checks omit execution-relevant values such as GEMM backend, cuBLASLt workspace/heuristics/autotune, attention mode and thresholds, chunk sizing, offload/residency policy, and compiled-program identity.

6. **Packed lifecycle validation has an unreachable branch.**  
   `eligible` rejects all phases other than `Ready` before separately testing `DecodePending`.

7. **Ragged prefill allocates every call.**  
   It creates a temporary `std::vector<uint8_t*>` and a temporary `DeviceBuffer<uint8_t*>` for flattened seen-token pointers.

8. **Ragged prefill projects vocabulary logits for all flattened tokens.**  
   If any request finalizes, final norm and LM head run over every flattened row before selected-row scattering.

9. **Ragged prefill timing does not measure GPU execution.**  
   `std::chrono` surrounds asynchronous launches without a CUDA completion event or synchronization before elapsed time is recorded.

10. **Host session state advances without an explicit GPU completion contract.**  
    Position and phase mutation occur before a formally modeled successful completion boundary.

11. **Persistent packed metadata invalidation is too weak.**  
    Session rebinding compares only owner pointers, so replaced cache storage or changed backing pointers may remain stale.

12. **The CPU backend still knows concrete checkpoint implementations.**  
    It includes GGUF/SafeTensors repository headers, performs `dynamic_cast<const GgufRepository*>`, and carries an architecture-owned naming-policy pointer into loading.

13. **Runtime composition remains hardcoded.**  
    Bootstrap uses process-static built-in catalogs and built-in architectures are centrally enumerated.

14. **`ResolvedModel` still overlaps responsibilities.**  
    It carries definition, graph, topology, weight plan, capabilities, identities/profiles, and a raw non-owning naming-policy pointer.

15. **Architecture-family implementations remain broad.**  
    Family `architecture.cpp` files typically combine probing, metadata decoding, validation, topology, graph, weight plan, capabilities, profile selection, and identity construction.

16. **The C API remains a large mapping and composition unit.**  
    `src/api/api.cpp` combines handles, error translation, ABI validation, option mapping, backend construction, engine/model/tokenizer APIs, and lifecycle entry points.

17. **Chat ownership improved, but valid states are not encoded.**  
    `ChatProfileCatalog` owns templates/codecs while `ChatCapabilities` still combines raw codec pointers and independent booleans.

18. **Native fan-out uses a manual begin/end protocol.**  
    `begin_native_fanout` and `end_native_fanout` depend on perfect caller pairing and source-buffer lifetime discipline.

19. **`GemmDispatcher` still combines too many responsibilities.**  
    It owns CUDA handles, Lt cache, autotuning, workspaces, all storage-layout dispatch, native activation quantization, and fan-out state.

20. **MMQ tensor-core capability is cached process-wide instead of per device.**  
    The default-on capability probe stores one static boolean based on the active device at first use. This is unsafe or at least underspecified if the process changes CUDA devices or uses heterogeneous GPUs. The fallback override also needs explicit test and diagnostic coverage.

21. **The MMQ default changed without changing the architectural contract.**  
    Tensor-core MMQ is now a default execution policy rather than an opt-in experiment. It therefore belongs in the compiled execution plan, diagnostics, compatibility fingerprint, benchmark matrix, and correctness gates.

22. **Some comments still describe central switch extension as OCP.**  
    A closed switch can be a valid performance design, but extending it is not Open/Closed compliance. Closed domains must be labeled honestly.

### 2.3 Current SOLID assessment

This is a static architectural score, not a runtime-quality score.

| Principle | Score | Main reason |
|---|---:|---|
| Single Responsibility | 4.5/10 | Packed execution, CPU weight setup, C API, serving composition, family resolution, and GEMM dispatch remain broad. |
| Open/Closed | 5.8/10 | Catalogs help, but built-in registration, bootstrap, chat profiles, backends, graph variants, and layouts still require central edits. |
| Liskov Substitution | 7.0/10 | Capability interfaces are mostly sound, but backend limits, role support, borrowed state, and completion semantics need stronger contracts. |
| Interface Segregation | 7.8/10 | Repository capabilities and public service views are narrow; internal context bags and multi-role implementations remain large. |
| Dependency Inversion | 6.4/10 | Neutral contracts are real progress, but CPU concrete casts, naming-policy leakage, static catalogs, and composition roots remain. |
| **Overall** | **~6.3/10** | Good direction with unresolved high-risk implementation concentrations. |

### 2.4 Target outcome

- **SRP:** 8.0+
- **OCP:** 8.0+
- **LSP:** 8.0+
- **ISP:** 8.5+
- **DIP:** 8.5+
- **Overall:** 8.3–8.7

The target does not require virtual dispatch in CUDA inner loops. The preferred runtime pattern remains:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

---

## 3. Non-Negotiable Rules

### 3.1 Preserve hot-path performance

Do not introduce deep virtual object graphs into transformer loops, attention launch loops, GEMM dispatch, MoE execution, sampling, or scheduler loops.

Use interfaces/registries at composition and compilation time. Execute immutable bindings, direct calls, compact tagged records, function pointers, or templates afterward.

### 3.2 Correctness before extraction

Do not modularize packed execution until the confirmed shape, allocation, completion, timing, and final-row issues have regression tests and fixes.

### 3.3 Separate policy from mechanism

Examples:

- checkpoint selection is policy; tensor reading is mechanism;
- architecture selection is policy; graph construction is mechanism;
- attention selection is policy; kernel launch is mechanism;
- weight layout is policy; linear execution is mechanism;
- MMQ tensor-core eligibility is policy; MMA/DP4A launch is mechanism;
- fan-out grouping is policy; activation quantization reuse is mechanism.

### 3.4 Use explicit capabilities

Keep optional behavior in separate, coherent contracts:

- repository location/random access/native block storage;
- vision encoding;
- tokenizer providers;
- chat/tool support;
- backend limits;
- asynchronous completion;
- MMQ/tensor-core capability;
- native fan-out support.

### 3.5 Avoid file-only refactoring

A valid SRP refactor changes ownership, lifetime, dependency direction, contracts, tests, and compilation boundaries. Moving methods without moving responsibility is not sufficient.

### 3.6 Declare closed domains

A switch is acceptable for an intentionally closed domain. Label it and test exhaustive handling.

```cpp
// CELEG_CLOSED_DOMAIN: LinearStorageKind
```

Do not describe adding a switch case as OCP compliance.

### 3.7 Treat device capability as device-scoped

CUDA capability caches must be keyed by device identity unless Celeg explicitly enforces one immutable CUDA device per process. Diagnostics must reveal the chosen device and policy.

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
        ├── token/numerical policies
        ├── model capabilities
        └── diagnostic provenance

ResolvedModel + backend options + device capabilities
    -> backend compiler
        -> immutable compiled program
        -> immutable execution plan
        -> compiled operator/linear bindings
        -> resource/workspace requirements
        -> compatibility fingerprint

Compiled backend
    -> sessions
    -> packed/non-packed executors
    -> scheduler adapters
    -> serving/protocol adapters
```

A backend must not require architecture-specific tensor-name generation or concrete checkpoint repository types. Runtime execution must not use architecture identity to select kernels.

---

## 5. Phase Overview

| Phase | Priority | Status | Goal |
|---|---:|---|---|
| 0 | P0 | **PARTIAL** | Finish correctness, allocation, completion, timing, and performance baselines. |
| 1 | P0 | **NOT STARTED** | Fix packed shape, lifecycle, allocation, final-row, timing, and commit defects. |
| 2 | P0 | **NOT STARTED** | Compile immutable CUDA plans and complete compatibility identity. |
| 3 | P0/P1 | **PARTIAL** | Finish checkpoint/backend inversion and resolve names before compilation. |
| 4 | P1 | **NOT STARTED** | Add injectable `RuntimeContext` and `RuntimeBuilder`. |
| 5 | P1 | **NOT STARTED** | Make resolved-model ownership authoritative. |
| 6 | P1 | **PARTIAL** | Split family resolution and registration responsibilities. |
| 7 | P1 | **BLOCKED** | Extract packed owned subsystems after Phase 1. |
| 8 | P0/P1 | **PARTIAL** | Refactor GEMM/layout dispatch, fan-out safety, and MMQ device policy. |
| 9 | P1 | **NOT STARTED** | Split C API mapping, handles, factories, and entry points. |
| 10 | P1/P2 | **PARTIAL** | Reduce serving duplication and make limits/modalities capability-driven. |
| 11 | P2 | **PARTIAL** | Make chat, codecs, roles, and tokenizers coherent and injectable. |
| 12 | P2 | **DECISION PENDING** | Declare graph operator families closed or registry-driven. |
| 13 | P2 | **PARTIAL** | Strengthen lifetime, LSP, completion, and device contracts. |
| 14 | P2 | **PARTIAL** | Expand architecture and hot-path boundary automation. |
| 15 | P0–P2 | **PARTIAL** | Continuously validate correctness, speed, memory, and compile cost. |
| 16 | P2/P3 | **REPLACED** | Follow the focused documentation strategy; do not recreate deleted docs blindly. |

---

# Part I — Safety and Packed Correctness

## 6. Phase 0 — Finish the Safety Net

### Status: PARTIAL

Already present:

- benchmark manifests and reproducibility tooling;
- numerical comparison helpers;
- compile measurement tooling;
- same-file native-GGUF comparison support;
- benchmark documentation;
- model-family manifests including MiniCPM5 and SmolLM3;
- existing native GGUF/MMQ correctness tests referenced by implementation comments.

Still required:

- packed vs non-packed decode parity;
- standard vs ragged prefill parity;
- synthetic `query_width != hidden` model;
- synthetic per-layer FFN-width model;
- mixed and zero-finalize ragged tests;
- allocation counters for host and device buffers;
- CUDA-event timing for packed paths;
- storage-generation rebinding tests;
- fan-out misuse/lifetime tests;
- MMQ tensor-core vs DP4A parity for production shapes;
- per-device capability tests, including device switching where supported;
- machine-readable results for every packed/GEMM milestone.

### Required matrix

| Dimension | Values |
|---|---|
| Backend | CPU, CUDA |
| Build | Debug, RelWithDebInfo, Release |
| Checkpoint | GGUF, SafeTensors, fake in-memory repository |
| Family | LFM2/LFM2.5, Granite, Gemma4, MiniCPM5, SmolLM3 |
| Weight | BF16, INT8, INT4, native Q4_K/Q6_K |
| KV | BF16, INT8 |
| Execution | single, packed decode, standard prefill, ragged prefill |
| Mixer | attention, convolution, hybrid |
| FFN | dense, MoE, variable dense widths |
| Attention | local, paged, segmented, shared KV, non-equal width |
| Native MMQ | DP4A, tensor core, explicit override on/off |
| CUDA device | each supported device; heterogeneous switching if permitted |

### Acceptance criteria

- every confirmed defect has a regression test before behavior changes;
- metrics separate host prepare, GPU execute, host commit, and end-to-end;
- steady-state allocations are measurable;
- benchmark artifacts identify model path/hash, build SHA, GPU/device index, driver, CUDA version, options, warmup, and repetitions;
- MMQ diagnostics state why tensor cores were selected or rejected.

---

## 7. Phase 1 — Fix Packed Execution Before Splitting It

### Status: NOT STARTED

### 7.1 Layer-specific attention width

Current defect:

```cpp
linear(op_output.data(), *attention.out, hidden.data(), rows,
       shape_.hidden, shape_.hidden, beta);
```

Target:

```cpp
const int query_width = attention.layout.query_width();
linear(op_output.data(), *attention.out, hidden.data(), rows,
       shape_.hidden, query_width, beta);
```

Compile and validate the value once in the packed layer binding.

### 7.2 Layer-specific dense FFN width

Required invariant:

```text
workspace capacity = topology.max_feed_forward_intermediate
layer width        = topology.feed_forward_intermediates[layer]
```

Suggested binding:

```cpp
struct PackedDenseFfnBinding {
    int intermediate = 0;
    const LinearWeight* w13 = nullptr;
    const LinearWeight* w2 = nullptr;
};
```

### 7.3 Centralize operation eligibility

```cpp
enum class PackedOperation { Decode, Prefill };

PackedEligibility validate_session(
    const PackedSessionContext& session,
    PackedOperation operation,
    const PackedExecutorCapabilities& capabilities);
```

Remove contradictory/unreachable phase checks.

### 7.4 Remove per-prefill allocations

Move into persistent workspace:

- flattened seen-token host/device pointers;
- finalized-row indices;
- gathered hidden and normalized rows;
- page-table expansion buffers;
- temporary metadata arrays.

No `DeviceBuffer` construction or capacity-growing vector is allowed in steady-state packed execution.

### 7.5 Gather final rows before final norm and LM head

Target sequence:

```text
execute transformer for all flattened tokens
build finalized-row indices
gather only finalized hidden rows
final norm only finalized rows
LM head only finalized rows
scale/softcap only finalized rows
scatter one logits row per finalized request
```

Non-finalizing rows advance position and remain `Prefilling` without vocabulary projection.

### 7.6 Define completion and commit semantics

```cpp
struct PackedTiming {
    double host_prepare_ms = 0.0;
    double gpu_execute_ms = 0.0;
    double host_commit_ms = 0.0;
    double end_to_end_ms = 0.0;
};

struct PackedCompletion {
    CudaEvent completed;
    PackedTiming timing;
};
```

Host phase/position/logit state must not be considered committed before successful completion under the public contract.

### 7.7 Strengthen metadata invalidation

Each session needs a storage generation or binding fingerprint covering every cached pointer table. Rebind when storage changes even if the owner object is unchanged.

### Acceptance criteria

- non-equal attention-width parity passes;
- variable FFN-width parity passes;
- no unreachable lifecycle branch remains;
- zero steady-state host/device allocation occurs;
- LM-head rows equal finalized request count;
- zero-finalize prefill performs no logits projection;
- CUDA timing correlates with profiler timing;
- failure cannot leave partially committed host state;
- metadata rebinds after cache replacement.

---

## 8. Phase 2 — Immutable Plans and Complete Compatibility

### Status: NOT STARTED

Compile execution policy when model options, device capability, and context limits become immutable.

```cpp
struct CudaPlanKey {
    int device_ordinal;
    int compute_capability;
    WeightMode weight_mode;
    KvCacheMode kv_cache_mode;
    GemmBackend gemm_backend;
    AttentionMode attention_mode;
    int attention_chunk_tokens;
    int attention_auto_threshold;
    bool fast_attention;
    bool fused_projections;
    bool fused_residuals;
    bool mmq_tensor_cores;
    size_t lt_workspace_bytes;
    int lt_heuristics;
    bool lt_autotune;
    uint64_t moe_policy_fingerprint;
    int max_context;
};
```

The actual key must include every value changing kernels, numerics, workspace, cache layout, offload/residency, fusion, or dispatch.

```cpp
struct PackedCompatibilityKey {
    const SharedModelWeights* weights = nullptr;
    uint64_t compiled_program_id = 0;
    uint64_t execution_plan_fingerprint = 0;
    int device_ordinal = -1;
    int max_context = 0;
};
```

### Acceptance criteria

- no plan compilation in steady-state decode/prefill;
- compilation count is visible in diagnostics;
- plan objects are immutable and thread-safe;
- packed batches reject different program/plan/device fingerprints;
- MMQ tensor-core selection is part of the plan, not ambient mutable state.

---

# Part II — Dependency Inversion and Neutral Contracts

## 9. Phase 3 — Finish the Checkpoint/Backend Boundary

### Status: PARTIAL

Completed foundation:

- neutral tensor/repository contracts;
- optional location and random-access capabilities;
- removal of direct GGUF flags from checkpoint/resolved model;
- family-owned naming-policy implementations.

Remaining violations:

- CPU backend concrete repository headers/casts;
- backend naming-policy use;
- raw naming-policy pointer in `ResolvedModel`;
- native-source decisions based on concrete type identity.

### Required changes

Model the actual native-storage need:

```cpp
class INativeBlockTensorRepository {
public:
    virtual ~INativeBlockTensorRepository() = default;
    virtual std::optional<NativeBlockTensorView> native_block_view(
        std::string_view resolved_name) const = 0;
};
```

Or encode it in a resolved request:

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

Resolve source names before backend compilation. Pack-cache policy must depend on resolved storage/layout capability, not repository class identity.

### Acceptance criteria

- backend directories include no concrete GGUF/SafeTensors repository headers;
- no backend casts to a concrete repository;
- no backend receives `ITensorNamingPolicy`;
- names/shapes/transforms resolve before compilation;
- fake repositories load through CPU and CUDA compiler tests;
- native GGUF performance and memory behavior remain validated.

---

## 10. Phase 4 — Injectable Runtime Composition

### Status: NOT STARTED

Existing catalogs support registration and freezing, but bootstrap creates process-static built-in catalogs and built-ins are centrally listed.

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

```cpp
ModelBootstrap load_model_bootstrap(
    const RuntimeContext& runtime,
    const std::filesystem::path& model_path);
```

A convenience default may delegate to one immutable built-in context.

### Acceptance criteria

- custom architecture/checkpoint tests edit no Celeg source;
- duplicate IDs fail explicitly;
- built-in registration is isolated;
- catalog lifetime/thread safety is documented;
- bootstrap, CLI, C API, and serving can receive a runtime context.

---

## 11. Phase 5 — Make `ResolvedModel` Authoritative

### Status: NOT STARTED

Target decomposition:

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

Rules:

- one authoritative owner per value;
- per-layer dimensions live in graph/program;
- maxima derive from per-layer data;
- compiled programs contain execution state only;
- provenance never drives backend execution;
- raw policy pointers are removed.

### Acceptance criteria

- no unexplained duplicated dimension/token/numerical state;
- execution does not branch on architecture/checkpoint identity;
- workspace derives from compiled program;
- lifetimes are value-owned, shared immutable, or stable-ID based.

---

# Part III — Families and Packed Subsystems

## 12. Phase 6 — Split Family Responsibilities

### Status: PARTIAL

Progress:

- major families have dedicated directories;
- naming policies are family-owned;
- tool-call codecs are increasingly family-owned;
- generic dense graph/weight-plan builders exist.

Target conceptual layout:

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

Not every family needs every file. The responsibility boundaries matter more than file count.

### Acceptance criteria

- metadata tests need no backend;
- graph tests need no checkpoint I/O;
- resolved-weight tests validate names/shapes independently;
- family registration is isolated;
- backend code contains no family switch.

---

## 13. Phase 7 — Extract Packed Owned Subsystems

### Status: BLOCKED by Phase 1

Target modules:

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

Ownership boundaries:

- `PackedWorkspace`: every reusable host/device buffer and capacity;
- `PackedBatchValidator`: pure host validation and structured errors;
- `PackedMetadataStager`: persistent binding, step/ragged staging, generations;
- `PackedCompiledProgram`: layer kinds, dimensions, weights, KV owner/slot, fusion, residual policy, compiled linear bindings;
- operator executors: narrow contexts, no lifecycle mutation;
- decode/prefill pipelines: orchestration only;
- completion/host commit: separate explicit step.

### Acceptance criteria

- no all-owning `PackedDecodeExecutorImpl` remains;
- transformer loop avoids repeated variant resolution;
- dimensions/bindings are immutable and prevalidated;
- no architecture identity in execution;
- no unexplained speed, latency, or VRAM regression.

---

# Part IV — CUDA Linear/MMQ Dispatch

## 14. Phase 8 — Refactor GEMM, Fan-Out, and MMQ Device Policy

### Status: PARTIAL

Current strengths:

- GEMM dispatch extracted from compiled model;
- cuBLAS/cuBLASLt/native GGUF paths centralized;
- Lt plan cache/autotuning exists;
- native activation fan-out reuse exists;
- MMQ tensor-core prefill implementation exists and is now default on detected `sm_72+`.

### 14.1 Make fan-out exception-safe

```cpp
class NativeFanoutScope {
public:
    NativeFanoutScope(GemmDispatcher&, const __nv_bfloat16* x, int m, int k);
    ~NativeFanoutScope();

    NativeFanoutScope(const NativeFanoutScope&) = delete;
    NativeFanoutScope& operator=(const NativeFanoutScope&) = delete;
};
```

Long-term target: compile a projection group that quantizes once and executes bound projections without ambient mutable scope state.

### 14.2 Make MMQ capability per-device

Replace one process-static boolean with either:

- a device-keyed immutable capability cache; or
- a documented/enforced single-device-per-process invariant.

Suggested contract:

```cpp
struct CudaDeviceCapabilities {
    int device_ordinal = -1;
    int compute_major = 0;
    int compute_minor = 0;
    bool int8_mma = false;
    bool mmq_tensor_core_supported = false;
};
```

Capability discovery must happen before plan compilation. Device changes after compilation must invalidate or reject the plan.

### 14.3 Treat tensor-core selection as compiled policy

The `CELEG_MMQ_TENSOR_CORES` override may remain a diagnostic/developer override, but the effective policy must be captured in:

- `CudaPlanKey`;
- plan fingerprint;
- diagnostics;
- benchmark artifact;
- packed compatibility;
- correctness matrix.

### 14.4 Separate collaborators

```text
CudaLinearRuntime
  ├── CublasHandles
  ├── LtPlanCache
  ├── LtAutotuner
  ├── LinearWorkspace
  ├── NativeQuantWorkspace
  ├── DeviceCapabilities
  └── compiled linear bindings
```

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

### Acceptance criteria

- fan-out cannot remain accidentally active;
- nested/mismatched/overwritten-source fan-out cases are tested;
- tensor-core policy is per-device and immutable in a plan;
- device switching cannot silently reuse the wrong capability;
- DP4A and tensor-core paths have parity tests and benchmark evidence;
- Lt cache/autotuner/workspaces are independently testable;
- new layouts do not edit packed operator code;
- comments correctly classify closed domains.

---

# Part V — API, Serving, Chat, and Graph

## 15. Phase 9 — Split the C API

### Status: NOT STARTED

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

Rules:

- preserve opaque handles/versioned structs;
- no exception crosses C boundary;
- add handle-kind tags and optional debug generations;
- backend construction uses factories/runtime context;
- ABI enum switches are mapping, not composition;
- mapping functions get table-driven tests.

### Acceptance criteria

- `api.cpp` is no longer a monolith;
- adding a backend does not edit unrelated APIs;
- CPU/CUDA mappings test independently;
- ABI compatibility remains green.

---

## 16. Phase 10 — Refactor Serving Composition

### Status: PARTIAL

Target roles:

```cpp
class IRequestService;
class IRequestResultStore;
class ISchedulerController;
class IServiceDiagnostics;
```

Required changes:

- extract backend-neutral request lifecycle;
- share CPU/CUDA orchestration where semantics match;
- publish backend/device limits;
- make vision provider-driven;
- remove profile-string and fixed mmproj conventions from generic composition;
- obtain tokenizers through providers;
- define cancellation/completion consistently.

### Acceptance criteria

- main composition roots are small/declarative;
- lifecycle duplication is reduced;
- no implementation silently narrows public contracts;
- modalities/tokenizers are capability/provider-driven.

---

## 17. Phase 11 — Coherent Chat and Tokenizer Composition

### Status: PARTIAL

Target registration:

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

Invalid combinations fail at registration.

### Acceptance criteria

- codec capability cannot be true while codec is null;
- unsupported roles fail through structured validation;
- profiles/tokenizer providers register through runtime context;
- adding a profile does not edit one central template file;
- serving does not cast repositories to build tokenizers.

---

## 18. Phase 12 — Declare the Graph Extension Model

### Status: DECISION PENDING

### Option A — Closed operator universe

Keep explicit variants and document that adding an operator requires coordinated graph/compiler/test changes.

### Option B — Compiler registry

Use stable operator IDs and compile-time registries while execution still receives concrete backend bindings.

### Recommendation

Use a two-level design:

1. small explicit neutral categories for stable common operators;
2. backend-specific compiled bindings;
3. postpone public operator plugins until a genuinely new family such as Mamba, DeltaNet, or RWKV proves the need.

### Acceptance criteria

- every operator domain is explicitly open or closed;
- unsupported backend/operator pairs fail during compilation;
- no architecture switch in execution loops;
- adding one synthetic operator has a measured change surface.

---

## 19. Phase 13 — Strengthen Contracts and Lifetimes

### Status: PARTIAL

Audit and formalize:

- packed borrowed pointers;
- storage invalidation generations;
- runtime-context catalog ownership;
- naming-policy lifetime until removed;
- chat codec ownership;
- `GemmDispatcher` options lifetime;
- fan-out scope lifetime;
- CUDA device selection/capability lifetime;
- service limits;
- asynchronous completion;
- cancellation/failure rollback.

### Acceptance criteria

- no undocumented narrower backend precondition;
- borrowed pointers document owner/mutation/invalidation/thread safety;
- async operations expose completion;
- device-scoped plans reject wrong-device use;
- raw policy pointers are removed or formally safe;
- CPU/CUDA contract tests share request semantics.

---

# Part VI — Governance, Performance, and Documentation

## 20. Phase 14 — Expand Automated Boundaries

### Status: PARTIAL

Add checks rejecting:

- concrete checkpoint repository headers/types in backends;
- architecture IDs in backend execution;
- profile-string branching outside registration/composition/diagnostics;
- `CudaExecutionPlan::compile` in decode/prefill/layer paths;
- `DeviceBuffer` construction or `cudaMalloc` in steady-state execution;
- unresolved naming policies in backends;
- manual fan-out without RAII/compiled group;
- process-global device capability caches unless single-device invariant is enforced;
- OCP claims for unannotated closed switches.

Add extension tests for custom architecture, checkpoint format, chat profile, tokenizer provider, backend factory, and fake repository.

### Plan maintenance rule

Every status change must include:

- commit SHA;
- tests added;
- benchmark artifact where runtime-sensitive;
- remaining deviations.

### Acceptance criteria

- CI rejects dependency-direction regressions;
- hot-path plan compilation/allocation regressions are caught;
- device-policy regressions are caught;
- public extension tests edit no built-in central file;
- completed tasks link to evidence.

---

## 21. Phase 15 — Continuous Performance Validation

### Status: PARTIAL

### Packed decode measurements

- batch 1, 2, 4, 8, 16, maximum;
- aggregate tokens/s;
- per-request p50/p95/p99;
- kernels/token;
- H2D/D2H bytes/token;
- synchronization count;
- host/GPU/commit timing;
- allocation count;
- VRAM/workspace.

### Ragged prefill shapes

```text
[16, 32, 64, 128]
[4, 512, 32, 2048]
[1024, 1024, 1024, 1024]
mixed finalize flags
all non-finalizing rows
```

Measure flattened tokens/s, finalized rows, actual LM-head rows, logits scratch, page-table volume, allocation count, and phase timing.

### MMQ measurements

For Q4_K and Q6_K:

- DP4A forced;
- tensor core forced;
- automatic selection;
- supported and unsupported compute capabilities;
- device switch if allowed;
- same-file/same-hash Celeg vs llama.cpp;
- decode and multiple prefill sizes;
- numerical parity/tolerance;
- workspace and activation-quantization reuse.

### Initial regression thresholds

| Metric | Allowed regression |
|---|---:|
| Correctness | zero unexplained |
| Decode throughput | <= 2% |
| Prefill throughput | <= 2% |
| p95 latency | <= 3% |
| VRAM | <= 2% unless explicit trade |
| Host allocation | zero in steady-state packed execution |
| Device allocation | zero in steady-state packed execution |

### Acceptance criteria

- every structural CUDA change has machine-readable comparison;
- event timing agrees with external profiling;
- same-file/same-hash rules are preserved;
- fan-out and tensor-core policy have correctness/performance evidence.

---

## 22. Phase 16 — Focused Documentation

### Status: REPLACED

Do not recreate deleted architecture documents automatically.

Near-term authoritative material:

- `README.md` — support and quick start;
- `API.md` — stable C API/ownership;
- `BENCHMARK.md` — reproducible methodology;
- this plan — architecture debt/status;
- `AGENTS.md` — contributor constraints;
- generated API diagnostics where appropriate.

Add only after stabilization:

- `docs/EXTENDING_CELEG.md` after runtime registration exists;
- `docs/PACKED_EXECUTION.md` after ownership/completion stabilize;
- `docs/BACKEND_COMPILATION.md` after resolved-weight/binding contracts stabilize;
- migration notes when public contracts change.

### Acceptance criteria

- docs describe implemented APIs, not planned names as existing;
- deleted snapshots are not recreated stale;
- extension examples compile in CI;
- packed/MMQ compatibility, timing, device policy, and workspace semantics are documented after implementation.

---

# Part VII — Commit Sequence

## 23. Recommended Milestones

### Milestone A — Packed shape correctness

1. Add non-equal attention-width fixture.
2. Add per-layer FFN-width fixture.
3. Fix attention output width.
4. Fix dense FFN width/workspace maximum.
5. Centralize lifecycle validation.
6. Add complete compatibility tests.

### Milestone B — Prefill completion and allocation

1. Add CUDA-event timing.
2. Define completion contract.
3. Move flattened pointer buffers into workspace.
4. Add allocation assertions.
5. Gather finalized rows.
6. Project only finalized rows.
7. Test mixed/zero finalize and rollback.

### Milestone C — Immutable plans and device policy

1. Define complete `CudaPlanKey`.
2. Add device-scoped capability object.
3. Compile plan during resource creation.
4. Add plan fingerprint.
5. Remove per-call compilation.
6. Include device/MMQ policy in compatibility.

### Milestone D — MMQ/fan-out safety

1. Add RAII fan-out guard.
2. Add fan-out lifecycle tests.
3. Replace static global MMQ capability with per-device policy.
4. Add DP4A/tensor-core parity tests.
5. Add automatic-selection diagnostics.
6. Benchmark automatic and forced modes.

### Milestone E — Checkpoint/backend boundary

1. Add neutral native-storage capability/policy.
2. Resolve tensor names before compilation.
3. Remove CPU concrete repository dependencies.
4. Remove backend naming policy.
5. Add fake repository tests.
6. Strengthen boundary checker.

### Milestone F — Runtime composition

1. Implement `RuntimeContext`.
2. Implement `RuntimeBuilder`.
3. Add family registration functions.
4. Inject runtime into bootstrap.
5. Inject into CLI/C API/serving.
6. Add public extension tests.

### Milestone G — Packed extraction

1. Extract workspace.
2. Extract validator/compatibility.
3. Extract metadata staging.
4. Compile layer bindings.
5. Extract operator executors.
6. Split decode/prefill pipelines.
7. Benchmark every extraction.

### Milestone H — Broader SRP cleanup

1. Make resolved model authoritative.
2. Split family stages.
3. Split C API.
4. Reduce serving duplication.
5. Complete chat/tokenizer providers.
6. Decide graph extension model.
7. Publish stabilized extension docs.

---

# Part VIII — Actionable Backlog

## 24. P0 Backlog

### Packed correctness

- [ ] Add synthetic `query_width != hidden` fixture.
- [ ] Add synthetic per-layer FFN-width fixture.
- [ ] Use `attention.layout.query_width()` for output projection input.
- [ ] Use current layer intermediate in dense FFN.
- [ ] Size workspace by maximum FFN intermediate.
- [ ] Add layer-indexed shape diagnostics.
- [ ] Centralize decode/prefill eligibility.
- [ ] Remove unreachable phase branch.

### Prefill and completion

- [ ] Move flat seen host/device storage into workspace.
- [ ] Add finalized-row index/gather storage.
- [ ] Normalize/project only gathered rows.
- [ ] Test mixed and zero finalize.
- [ ] Add CUDA event timing.
- [ ] Define host commit boundary.
- [ ] Test failure before commit.
- [ ] Instrument host/device allocations.
- [ ] Assert zero steady-state allocation.

### Plans and compatibility

- [ ] Define complete `CudaPlanKey`.
- [ ] Define compiled-program ID.
- [ ] Define stable fingerprint.
- [ ] Include device/MMQ policy.
- [ ] Replace partial compatibility checks.
- [ ] Remove per-call plan compile/set/reset.
- [ ] Expose plan compile count.

### MMQ and fan-out

- [ ] Add per-device `CudaDeviceCapabilities`.
- [ ] Remove process-global capability assumption.
- [ ] Define device-switch behavior.
- [ ] Capture effective MMQ policy in diagnostics.
- [ ] Test DP4A vs tensor-core parity.
- [ ] Test automatic override behavior.
- [ ] Add RAII fan-out scope.
- [ ] Test fan-out mismatch/nesting/source overwrite.

---

## 25. P1 Backlog

### Checkpoint/backend boundary

- [x] Extract neutral tensor contracts.
- [x] Extract optional repository capabilities.
- [x] Remove direct GGUF flags from checkpoint/resolved model.
- [x] Move family naming policies out of generic roles.
- [ ] Model native block storage neutrally.
- [ ] Resolve source names before backend compilation.
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
- [ ] Add public extension tests.

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
- [ ] Split decode/prefill pipelines.

### GEMM/layout dispatch

- [x] Extract `GemmDispatcher`.
- [x] Add Lt cache/autotune.
- [x] Add native GGUF paths.
- [x] Add activation fan-out reuse.
- [x] Add MMQ tensor-core path and automatic enablement.
- [ ] Make device capability explicit/per-device.
- [ ] Replace manual fan-out scope.
- [ ] Separate handles/cache/autotuner/workspaces.
- [ ] Compile linear bindings.
- [ ] Correct closed-domain/OCP comments.

### API and serving

- [ ] Split C API modules.
- [ ] Add backend factory catalog.
- [ ] Extract shared request lifecycle.
- [ ] Publish backend/device limits.
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

### Families

- [x] Dedicated family directories.
- [x] MiniCPM5 support.
- [x] SmolLM3 support.
- [x] Several family naming policies/tool codecs split.
- [ ] Split probe/metadata/topology/graph/weights/capabilities.
- [ ] Add registration functions.
- [ ] Add stage-focused tests.

### Chat/tokenizer

- [x] Conversation/generation abstractions.
- [x] Family tool-call codecs.
- [x] Catalog-owned templates/codecs.
- [ ] Replace raw codec pointer in capabilities.
- [ ] Encode role capabilities.
- [ ] Add tokenizer providers.
- [ ] Register through runtime context.

### Governance/docs

- [x] Benchmark manifests/numerical helpers.
- [x] Focused API/benchmark guides.
- [ ] Packed allocation/plan boundary checks.
- [ ] Device-capability boundary checks.
- [ ] Public extension compilation tests.
- [ ] Evidence links for status transitions.
- [ ] Create extension/packed docs only after interfaces stabilize.

---

# Part IX — Definition of Done

## 27. Architectural Definition of Done

The refactoring is complete when:

- runtime composition is injectable;
- built-in families register without central runtime switches;
- backends do not depend on concrete checkpoint formats;
- names/transforms resolve before backend compilation;
- resolved-model values have one authoritative owner;
- compiled programs contain execution state only;
- optional behavior is capability-driven;
- CUDA device policy is explicit and immutable per compiled plan;
- raw borrowed policy lifetimes are removed or formally safe;
- graph domains are explicitly open or closed;
- C API/serving roots are small and declarative;
- boundary CI enforces dependency direction.

## 28. Packed Execution Definition of Done

Packed execution is complete when:

- non-equal attention and variable FFN widths pass;
- no per-call CUDA plan compilation occurs;
- compatibility uses complete immutable fingerprints;
- workspace is dedicated;
- validation is pure host-side;
- metadata staging is generation-aware;
- layer dimensions/bindings compile once;
- operators are separated;
- decode/prefill are separate orchestration pipelines;
- prefill projects only finalized rows;
- zero steady-state host/device allocation occurs;
- metrics measure actual GPU work;
- completion/commit semantics are explicit;
- failure leaves no partial state;
- performance remains within thresholds.

## 29. CUDA Linear/MMQ Definition of Done

The linear/MMQ refactor is complete when:

- fan-out lifetime is exception-safe;
- MMQ capability is per-device or a single-device invariant is enforced;
- effective tensor-core policy is compiled and fingerprinted;
- wrong-device plan reuse is impossible;
- DP4A/tensor-core parity is continuously tested;
- automatic policy is diagnosable;
- dispatcher responsibilities are separated without hot-path regression;
- layout extension behavior is honestly documented.

## 30. Final Priority

1. **Fix packed shape defects.**
2. **Fix prefill allocation, final-row projection, completion, and timing.**
3. **Compile immutable plans with device/MMQ policy.**
4. **Make MMQ capability per-device and fan-out exception-safe.**
5. **Finish checkpoint/backend dependency inversion.**
6. **Introduce runtime composition.**
7. **Extract packed ownership boundaries.**
8. **Make resolved-model ownership authoritative.**
9. **Split architecture, C API, serving, and chat responsibilities.**
10. **Stabilize and document extension contracts.**

The project should optimize for clear ownership, explicit device policy, and precompiled execution rather than merely shorter source files.
