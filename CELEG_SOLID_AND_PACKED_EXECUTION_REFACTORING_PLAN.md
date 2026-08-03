# Celeg SOLID and Packed CUDA Execution Refactoring Plan

**Project:** `celsowm/celeg`  
**Primary scope:** architecture, extensibility, dependency boundaries, runtime contracts, CUDA packed decode, ragged prefill, and hot-path maintainability  
**Baseline:** static source review of the current `master` branch around commit `8907e01cadede5c1c28426d2362e9effb9a53a53`  
**Important limitation:** this plan is based on static analysis. Build, correctness, benchmark, Nsight Systems, Nsight Compute, sanitizer, and hardware validation must be performed during implementation.

---

## 1. Executive Summary

Celeg already has a substantially better architecture than a typical early-stage inference runtime. It has:

- explicit architecture resolution;
- checkpoint format abstractions;
- a neutral resolved-model representation;
- CPU and CUDA backend separation;
- backend compiler concepts;
- capability-oriented interfaces;
- focused public diagnostic, persistence, and session views;
- architecture boundary documentation;
- a structural boundary checker;
- an OpenAI-compatible serving layer;
- support for multiple model families and checkpoint formats.

The project is therefore not structurally broken. Its current problem is that the intended architecture is more mature than several important implementation areas.

The main architectural gaps are:

1. **Composition is still hardcoded.**
   Built-in architectures, checkpoint formats, chat profiles, tokenizers, API backends, and serving behavior are selected in central factories or composition roots.

2. **Backend code still leaks checkpoint-specific knowledge.**
   CPU weight materialization knows concrete GGUF and SafeTensors repository types, weakening dependency inversion.

3. **Resolved-model state is too broad and duplicated.**
   `ModelDefinition`, `RuntimeTopology`, `ResolvedModel`, and `CompiledModelProgram` overlap in responsibilities and metadata.

4. **Architecture modules have excessive responsibilities.**
   A model architecture implementation often performs detection, metadata decoding, validation, graph construction, topology construction, weight planning, capability definition, tensor naming, and chat-profile selection.

5. **The C API and serving composition roots are oversized.**
   They contain mapping, validation, construction, dispatch, backend selection, tokenizer selection, vision handling, lifecycle, and protocol logic.

6. **Closed variants and switches are not consistently treated as closed domains.**
   Some are legitimate and performance-oriented; others are extension bottlenecks incorrectly described as Open/Closed-compliant.

7. **`packed_execution.cu` is the most severe SRP hotspot in the CUDA backend.**
   It combines workspace ownership, metadata staging, validation, sampling, embedding, attention, KV management, convolution, dense FFN, MoE, decode, ragged prefill, session mutation, and metrics.

8. **The packed execution path contains correctness and performance risks.**
   Examples include global-dimension assumptions, incomplete compatibility checks, per-call execution-plan compilation, avoidable allocations, and prefill timing that may not measure actual GPU execution.

### Baseline SOLID assessment

| Principle | Baseline score | Main reason |
|---|---:|---|
| Single Responsibility | 4.0/10 | Oversized architecture implementations, C API, serving roots, CPU setup, GEMM dispatcher, and especially `packed_execution.cu` |
| Open/Closed | 5.5/10 | Backends are mostly model-neutral, but registration, chat, formats, graph variants, quantization dispatch, and packed execution require central edits |
| Liskov Substitution | 7.0/10 | Most capability interfaces are sound, but some interfaces expose broader contracts than implementations consistently support |
| Interface Segregation | 8.0/10 | Strong public views and capability interfaces, weakened by large internal context bags and service objects implementing several roles |
| Dependency Inversion | 6.0/10 | Good abstractions exist, but static catalogs, concrete repository dependencies, raw policy pointers, and hardcoded composition reduce effective inversion |
| **Overall** | **~6.4/10** | Good architectural direction with several high-impact implementation hotspots |

### Target outcome

After completing the high-priority phases in this plan, Celeg should reach approximately:

- **SRP:** 8.0+
- **OCP:** 8.0+
- **LSP:** 8.0+
- **ISP:** 9.0
- **DIP:** 8.5+
- **Overall:** approximately 8.3–8.7

The target is not to introduce virtual dispatch into CUDA hot paths. The target is to move policy decisions and binding to construction or compilation time, then execute compact, concrete plans.

---

## 2. Refactoring Principles

All implementation work should follow these rules.

### 2.1 Preserve hot-path performance

Do not replace direct calls with deep virtual object graphs inside:

- transformer layer loops;
- GEMM dispatch loops;
- attention kernel launches;
- sampling loops;
- MoE expert execution;
- decode scheduler loops.

Prefer:

- construction-time binding;
- immutable execution plans;
- function pointers or compact tagged records where appropriate;
- precomputed dimensions and offsets;
- arrays of layer bindings;
- templates only where compile-time specialization is valuable;
- contiguous workspace ownership.

### 2.2 Separate policy from mechanism

Examples:

- **Policy:** which checkpoint format is selected.
- **Mechanism:** reading bytes from a tensor repository.

- **Policy:** which attention implementation should be used.
- **Mechanism:** launching a concrete CUDA kernel.

- **Policy:** which weight layout is selected.
- **Mechanism:** embedding lookup or linear execution.

- **Policy:** whether segmented attention is required.
- **Mechanism:** workspace sizing and kernel launch.

Policy should be resolved before the hot path whenever possible.

### 2.3 Prefer capabilities over flag combinations

Avoid combinations such as:

```cpp
supports_tools = true;
tool_codec = nullptr;
```

Prefer coherent capability objects where invalid states are unrepresentable.

### 2.4 Make extension points explicit

There should be a documented answer to each question:

- How does an external application add a checkpoint format?
- How does it add a model architecture?
- How does it add a chat profile?
- How does it add a tokenizer provider?
- How does it add a backend?
- How does it add a new operator family?
- How does it add a new CUDA weight layout?
- Which extensions require rebuilding Celeg?
- Which extensions can be injected by a consumer?

### 2.5 Avoid file-only refactoring

Splitting one 1,100-line file into ten files without separating ownership and responsibilities does not solve SRP.

A valid refactoring must change:

- ownership;
- public and internal contracts;
- dependency direction;
- lifecycle boundaries;
- testability;
- compilation boundaries;
- extension behavior.

### 2.6 Add characterization tests before behavior changes

Before refactoring complex execution code:

1. capture current behavior;
2. add deterministic fixtures;
3. add parity tests;
4. record benchmark baselines;
5. refactor;
6. compare correctness and performance.

---

## 3. Architectural Target

The intended high-level dependency chain should remain:

```text
checkpoint formats
    -> architecture resolution
        -> neutral resolved model
            -> backend compiler
                -> backend execution
                    -> serving/protocol adapters
```

The concrete target should be:

```text
RuntimeContext / RuntimeBuilder
    ├── CheckpointFormatCatalog
    ├── ArchitectureCatalog
    ├── ChatProfileCatalog
    ├── TokenizerProviderCatalog
    ├── BackendFactoryCatalog
    ├── DiagnosticsSink
    └── Optional application policies

CheckpointSource
    -> ICheckpointFormat
        -> IWeightRepository
        -> CheckpointMetadataView

CheckpointMetadataView
    -> IArchitecture
        -> ResolvedModelDefinition
        -> ModelGraph
        -> WeightPlan
        -> ModelCapabilities
        -> TextProfileReference

ResolvedModel
    -> IBackendCompiler
        -> CompiledModelProgram
        -> Backend-specific immutable bindings
        -> Backend workspace/resource factories

Compiled backend
    -> inference sessions
    -> schedulers
    -> serving adapters
```

### Key rule

A backend should receive resolved, format-neutral data. It should not need to ask:

- whether the source was GGUF;
- whether the source was SafeTensors;
- which architecture produced it;
- how architecture-specific tensor names are generated;
- which chat profile was selected;
- which repository concrete type is in use.

Diagnostic metadata may preserve this information separately, but executable code should not depend on it.

---

## 4. Phase Overview

| Phase | Priority | Goal |
|---|---:|---|
| 0 | P0 | Establish correctness and performance baselines |
| 1 | P0 | Fix packed execution correctness risks |
| 2 | P0 | Introduce injectable runtime composition |
| 3 | P0 | Remove checkpoint-format knowledge from backends |
| 4 | P1 | Redesign resolved-model contracts and ownership |
| 5 | P1 | Split architecture modules by responsibility |
| 6 | P1 | Refactor `packed_execution.cu` into explicit subsystems |
| 7 | P1 | Compile immutable packed execution plans |
| 8 | P1 | Refactor CUDA GEMM and weight-layout dispatch |
| 9 | P1 | Refactor C API and backend construction |
| 10 | P1 | Refactor serving composition and service roles |
| 11 | P2 | Refactor chat profiles, templates, codecs, and tokenizer composition |
| 12 | P2 | Reassess graph extensibility and operator families |
| 13 | P2 | Strengthen LSP, capability, and lifetime contracts |
| 14 | P2 | Expand architecture-boundary automation |
| 15 | P2 | Performance validation and cleanup |
| 16 | P3 | Documentation, migration guides, and extension examples |

Phases 0–3 should be completed before broad structural experimentation. Phase 6 must not start without Phase 0 characterization coverage.

---

# Part I — Baseline and Safety

## 5. Phase 0 — Establish Baselines

### Objective

Create a reproducible reference for correctness, compatibility, memory usage, and performance before changing architecture or CUDA execution.

### 5.1 Build matrix

Create CI and local scripts covering at least:

| Dimension | Values |
|---|---|
| Backend | CPU, CUDA |
| Build type | Debug, RelWithDebInfo, Release |
| Checkpoint | GGUF, SafeTensors where supported |
| Model family | LFM2/LFM2.5, Granite, Gemma4, MiniCPM5 |
| Weight mode | BF16, INT8, INT4, supported GGUF layouts |
| KV mode | BF16, INT8 |
| Execution | single-session, packed decode, ragged prefill |
| Attention | normal, paged, segmented where available |
| FFN | dense, MoE |
| Mixer | attention, short convolution |

### 5.2 Deterministic correctness tests

For each supported model shape:

- fixed prompt;
- fixed RNG seed;
- greedy decoding where possible;
- fixed generation settings;
- expected token sequence or logits checksum;
- single-session vs packed decode parity;
- normal prefill vs ragged prefill parity;
- local KV vs paged KV parity;
- BF16 KV vs INT8 KV tolerance;
- fused vs unfused projection parity;
- fused vs unfused residual parity.

### 5.3 Shape-focused synthetic models

Add tiny synthetic fixtures where:

1. `query_width == hidden`;
2. `query_width != hidden`;
3. FFN intermediate is constant;
4. FFN intermediate differs by layer;
5. shared KV owner layers are used;
6. short convolution and attention alternate;
7. dense and MoE layers coexist;
8. sliding-window attention is enabled;
9. final logit softcap is enabled;
10. embedding and logits weights use different layouts if supported.

These tests are critical because normal production models may accidentally hide generic-shape defects.

### 5.4 Performance baseline

Record:

- prefill tokens/s;
- decode tokens/s;
- packed decode aggregate tokens/s;
- per-request latency;
- p50/p95/p99 latency;
- scheduler overhead;
- CUDA kernel count per token;
- host-to-device copy count per token;
- device allocation count during steady state;
- cuBLAS/cuBLASLt heuristic calls during steady state;
- VRAM peak;
- workspace size;
- CPU-side enqueue time;
- actual GPU elapsed time using CUDA events;
- Nsight Systems timeline;
- Nsight Compute metrics for key kernels.

### 5.5 Required artifacts

Create:

```text
benchmarks/baselines/
  hardware.json
  build.json
  correctness.json
  performance.json
  nsight-systems/
  nsight-compute/
```

### Acceptance criteria

- Reproducible build scripts exist.
- Packed and non-packed parity is tested.
- At least one synthetic nontraditional shape is tested.
- No steady-state CUDA allocation goes unmeasured.
- Baseline benchmark results are stored in machine-readable format.
- CI can detect correctness regressions.
- Performance comparison scripts tolerate normal noise and report significant regressions.

---

# Part II — Immediate Packed Execution Corrections

## 6. Phase 1 — Fix Packed Execution Correctness Risks

### Scope

Primary file:

```text
src/backend/cuda/model/packed_execution.cu
```

Related contracts:

```text
include/celeg/backend/cuda/packed.hpp
include/celeg/backend/cuda/packed_session.hpp
include/celeg/model/graph.hpp
include/celeg/model/resolved.hpp
include/celeg/backend/cuda/execution_plan.hpp
```

### 6.1 Use layer-specific attention dimensions

Current packed attention code appears to assume that the attention output projection input width equals `hidden`.

Replace global assumptions with values derived from the concrete `AttentionSpec` or a compiled packed-layer binding.

Required fields should include:

```cpp
struct PackedAttentionBinding {
    int query_width;
    int key_value_width;
    int query_heads;
    int key_value_heads;
    int head_dim;
    int output_width;
    int sliding_window;
    int kv_owner_layer;
};
```

The output projection must use the actual operator input width:

```cpp
linear(
    attention_output,
    *attention.out,
    hidden,
    rows,
    topology.hidden,
    binding.query_width,
    beta
);
```

### 6.2 Use layer-specific FFN dimensions

Do not use one global `shape_.intermediate` for every dense FFN layer.

Compile:

```cpp
struct PackedDenseFfnBinding {
    int intermediate;
    bool fused_w13;
    const LinearWeight* w13;
    const LinearWeight* w1;
    const LinearWeight* w3;
    const LinearWeight* w2;
};
```

Workspace capacity should use:

```text
maximum dense intermediate across the compiled program
```

Execution should use:

```text
current layer intermediate
```

### 6.3 Validate all execution-relevant compatibility

Replace manual field-by-field partial checks with a stable fingerprint.

Example:

```cpp
struct PackedExecutionCompatibilityKey {
    const SharedModelWeights* weights;
    uint64_t compiled_program_id;
    uint64_t execution_plan_hash;
    int max_context;
};
```

The execution-plan hash must include every option that changes:

- kernel selection;
- workspace requirements;
- numerical behavior;
- caching behavior;
- MoE residency behavior;
- attention behavior;
- GEMM dispatch;
- KV representation;
- fusion behavior.

### 6.4 Eliminate unreachable lifecycle checks

Centralize session-phase validation.

Example:

```cpp
enum class PackedOperation {
    Decode,
    Prefill
};

PackedEligibilityResult validate_phase(
    SessionPhase phase,
    PackedOperation operation
);
```

Avoid independent condition chains that can become contradictory.

### 6.5 Correct prefill timing

Use CUDA events for GPU execution timing.

Keep separate metrics:

```cpp
struct PackedTiming {
    double host_prepare_ms;
    double gpu_execute_ms;
    double host_commit_ms;
    double end_to_end_ms;
};
```

Do not describe host enqueue time as prefill execution time.

### 6.6 Add assertions for compiled shape consistency

At construction or compile time, validate:

- all weight rows and columns;
- query, key, and value projection widths;
- attention output projection width;
- dense FFN widths;
- MoE widths;
- workspace maximums;
- cache layer ownership;
- paged-KV slot mapping;
- convolution state width.

### Acceptance criteria

- Packed execution passes synthetic `query_width != hidden` tests.
- Packed execution passes variable-FFN-width tests.
- Compatibility checks reject sessions with different execution plans.
- CUDA-event timing matches external profiler measurements within expected tolerance.
- Lifecycle validation contains no unreachable branch.
- No shape decision in the layer loop depends on architecture identity.

---

# Part III — Dependency Inversion and Composition

## 7. Phase 2 — Introduce `RuntimeContext` and `RuntimeBuilder`

### Problem

Celeg has abstract catalogs, but built-in catalogs are created through static functions and used globally. This limits:

- consumer injection;
- test isolation;
- custom architecture support;
- custom checkpoint support;
- replacement of chat/tokenizer policies;
- deterministic composition.

### Target API

```cpp
class RuntimeContext {
public:
    const ArchitectureCatalog& architectures() const;
    const CheckpointFormatCatalog& checkpoint_formats() const;
    const ChatProfileCatalog& chat_profiles() const;
    const TokenizerProviderCatalog& tokenizer_providers() const;
    const BackendFactoryCatalog& backends() const;
};

class RuntimeBuilder {
public:
    RuntimeBuilder& add_builtin_components();
    RuntimeBuilder& add_architecture(
        std::unique_ptr<IArchitecture> architecture);
    RuntimeBuilder& add_checkpoint_format(
        std::unique_ptr<ICheckpointFormat> format);
    RuntimeBuilder& add_chat_profile(
        ChatProfile profile);
    RuntimeBuilder& add_tokenizer_provider(
        std::unique_ptr<ITokenizerProvider> provider);
    RuntimeBuilder& add_backend_factory(
        std::unique_ptr<IBackendFactory> factory);

    RuntimeContext build();
};
```

### Rules

- No process-global mutable catalogs.
- Built-ins are a composition convenience, not the only path.
- Tests can construct minimal contexts.
- Catalog freeze should occur when `RuntimeContext` is built.
- Registration IDs must be unique.
- Duplicate registration must produce explicit errors.
- Runtime context lifetime must outlive objects that borrow catalog-owned data, or ownership must be transferred safely.

### Migration

Replace:

```cpp
load_model_bootstrap(source)
```

with:

```cpp
load_model_bootstrap(runtime_context, source)
```

Optionally retain a convenience overload:

```cpp
load_model_bootstrap(source)
```

that delegates to an immutable default built-in context.

### Acceptance criteria

- A test can register a fake architecture without editing Celeg source.
- A test can register a fake checkpoint format without editing Celeg source.
- Built-in registration is isolated in one composition module.
- No internal code depends on a mutable singleton catalog.
- Public documentation explains lifetime and thread-safety.

---

## 8. Phase 3 — Remove Concrete Checkpoint Knowledge from Backends

### Problem

Backend weight code currently knows concrete repository implementations such as GGUF and SafeTensors and uses concrete-type checks.

This violates the intended dependency direction:

```text
checkpoint format -> neutral model -> backend
```

### Target abstraction

Extend capability interfaces rather than using concrete casts.

Possible contracts:

```cpp
class IWeightRepository {
public:
    virtual ~IWeightRepository() = default;
    virtual TensorDescriptor describe(std::string_view name) const = 0;
    virtual void read(
        std::string_view name,
        MutableByteSpan destination) const = 0;
};

class INativeTensorViewRepository {
public:
    virtual ~INativeTensorViewRepository() = default;
    virtual std::optional<NativeTensorView> native_view(
        std::string_view name) const = 0;
};

class IRandomAccessTensorReader {
public:
    virtual ~IRandomAccessTensorReader() = default;
    virtual void read_range(
        std::string_view name,
        uint64_t offset,
        MutableByteSpan destination) const = 0;
};
```

The backend may depend on capabilities, not concrete formats.

### Resolve tensor names before backend compilation

Current backends should not receive an architecture-specific naming policy and construct names at runtime.

Change `WeightPlan` from requests such as:

```text
logical tensor request + naming policy
```

to:

```text
fully resolved source tensor name
+ destination semantic
+ expected shape
+ layout conversion
+ optional slicing/transform
```

Example:

```cpp
struct ResolvedWeightRequest {
    WeightSemantic semantic;
    std::string source_tensor_name;
    TensorShape expected_source_shape;
    TensorShape expected_destination_shape;
    WeightTransform transform;
    LayerIndex layer;
};
```

### Diagnostic metadata

Preserve format information separately:

```cpp
struct ModelProvenance {
    std::string architecture_id;
    std::string checkpoint_format_id;
    std::string source_description;
};
```

Execution code should not branch on it.

### Acceptance criteria

- No backend includes concrete GGUF repository headers.
- No backend includes concrete SafeTensors repository headers.
- No backend uses `dynamic_cast<GgufRepository*>`.
- No backend uses `dynamic_cast<SafeTensorRepository*>`.
- Tensor names are fully resolved before backend compilation.
- A synthetic checkpoint repository can load a model through CPU and CUDA tests.

---

# Part IV — Neutral Model Contracts

## 9. Phase 4 — Redesign `ResolvedModel` and Related Types

### Problems

`ModelDefinition`, `RuntimeTopology`, `ResolvedModel`, and `CompiledModelProgram` overlap.

Potential duplication includes:

- hidden size;
- layer count;
- head dimensions;
- token IDs;
- numerical scales;
- architecture identity;
- source format;
- per-layer values;
- capability state.

This creates ambiguity:

- Which object is authoritative?
- Which fields are runtime requirements?
- Which fields are diagnostics?
- Which values are global defaults versus per-layer values?
- Which values survive backend compilation?

### Target decomposition

```cpp
struct ModelIdentity {
    std::string architecture_id;
    std::string model_profile_id;
};

struct ModelDimensions {
    int vocab_size;
    int hidden;
    int layers;
    int maximum_attention_projection_width;
    int maximum_attention_head_dim;
    int maximum_dense_intermediate;
    int maximum_moe_intermediate;
};

struct NumericalPolicy {
    float norm_eps;
    float embedding_multiplier;
    float residual_multiplier;
    float logits_divisor;
    float final_logit_softcap;
};

struct TokenPolicy {
    std::optional<int32_t> bos;
    std::vector<int32_t> eos;
    std::optional<int32_t> pad;
};

struct ModelCapabilities {
    AttentionCapabilities attention;
    MixerCapabilities mixers;
    FfnCapabilities ffn;
    ModalCapabilities modalities;
    TextCapabilities text;
};

struct ResolvedModel {
    ModelIdentity identity;
    ModelDimensions dimensions;
    NumericalPolicy numerics;
    TokenPolicy tokens;
    ModelGraph graph;
    ResolvedWeightPlan weights;
    ModelCapabilities capabilities;
    ModelProvenance provenance;
};
```

### Compiled program

`CompiledModelProgram` should contain only execution-relevant state.

Do not require:

- checkpoint source format;
- architecture ID for execution decisions;
- unresolved tensor naming policies;
- metadata probing state.

Diagnostic identity can be attached in a sidecar object.

### Ownership

Eliminate ambiguous raw ownership comments.

Use one of:

- value ownership;
- `shared_ptr<const T>`;
- runtime-context-owned immutable object with explicit documented lifetime;
- stable interned IDs rather than raw policy pointers.

### Acceptance criteria

- Each field has one authoritative owner.
- Execution program contains no checkpoint-format branching data.
- All per-layer widths are represented explicitly.
- Maximum workspace dimensions are derived and validated.
- Diagnostic provenance is available without affecting execution.
- Raw pointer lifetime assumptions are either removed or formally documented and tested.

---

# Part V — Architecture Modules

## 10. Phase 5 — Split Architecture Implementations

### Current concern

Architecture files often perform all of the following:

- architecture detection;
- repository hint processing;
- metadata key lookup;
- JSON parsing;
- GGUF metadata parsing;
- shape validation;
- token extraction;
- topology construction;
- graph construction;
- weight-plan construction;
- tensor naming;
- chat-profile assignment;
- capability assignment.

This creates many reasons to modify one file.

### Target layout

For each model family:

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
  text_profile.cpp
  tensor_names.cpp
  architecture.cpp
```

Not every family needs every file, but the conceptual boundaries should remain.

### Suggested contracts

```cpp
class IArchitectureProbe {
public:
    virtual ProbeResult probe(
        const CheckpointMetadataView& metadata) const = 0;
};

class IArchitectureResolver {
public:
    virtual ResolvedModel resolve(
        const ArchitectureResolutionInput& input) const = 0;
};
```

Internally, a family can use concrete functions rather than virtual interfaces:

```cpp
FamilyMetadata decode_metadata(...);
void validate_metadata(const FamilyMetadata&);
ModelGraph build_graph(const FamilyMetadata&);
ResolvedWeightPlan build_weight_plan(
    const FamilyMetadata&,
    const ModelGraph&);
```

### Registration

A family registration function should register:

- probe/resolver;
- chat profile;
- tokenizer provider if family-specific;
- optional multimodal components.

Example:

```cpp
void register_minicpm5(RuntimeBuilder& builder);
```

Adding a family should not require editing several central switch files.

### Acceptance criteria

- Adding a test architecture requires no change to a central built-in factory.
- Architecture resolution functions are independently unit-testable.
- Metadata parsing tests do not require CUDA or CPU backend construction.
- Graph tests do not require checkpoint I/O.
- Weight-plan tests validate source names and shapes independently.
- Chat-profile selection is not embedded in backend code.

---

# Part VI — Packed CUDA Execution

## 11. Phase 6 — Refactor `packed_execution.cu`

### Current responsibility inventory

`PackedDecodeExecutorImpl` currently combines:

1. executor capacity validation;
2. stream ownership;
3. cuBLAS ownership;
4. GEMM dispatcher ownership;
5. active execution-plan ownership;
6. persistent GPU workspace;
7. persistent pinned-host workspace;
8. segmented attention scratch growth;
9. session compatibility checks;
10. decode eligibility;
11. prefill eligibility;
12. session-layout change detection;
13. persistent pointer-table assembly;
14. per-step metadata assembly;
15. page-table flattening;
16. ragged row flattening;
17. host-to-device copies;
18. embedding layout dispatch;
19. linear shape validation;
20. QKV projection;
21. Q/K normalization and RoPE;
22. paged KV store;
23. local KV store;
24. BF16 KV execution;
25. INT8 KV execution;
26. segmented attention;
27. standard attention;
28. shared-KV owner resolution;
29. short convolution;
30. dense FFN;
31. fused and unfused projections;
32. fused and unfused residuals;
33. MoE routing;
34. expert residency;
35. MoE execution;
36. transformer layer loop;
37. sampling;
38. decode pipeline;
39. ragged prefill pipeline;
40. logits projection;
41. session-state scatter;
42. host session mutation;
43. timing;
44. metrics.

This is the largest SRP violation in the CUDA model layer.

### Target directory

```text
src/backend/cuda/model/packed/
  types.hpp
  compatibility.hpp
  compatibility.cpp
  workspace.hpp
  workspace.cu
  metadata_stager.hpp
  metadata_stager.cu
  attention_workspace.hpp
  attention_workspace.cu
  layer_program.hpp
  layer_program.cpp
  embedding_executor.hpp
  embedding_executor.cu
  attention_executor.hpp
  attention_executor.cu
  convolution_executor.hpp
  convolution_executor.cu
  dense_ffn_executor.hpp
  dense_ffn_executor.cu
  moe_executor.hpp
  moe_executor.cu
  transformer_executor.hpp
  transformer_executor.cu
  final_projection.hpp
  final_projection.cu
  decode_pipeline.hpp
  decode_pipeline.cu
  prefill_pipeline.hpp
  prefill_pipeline.cu
  metrics.hpp
  executor.cpp
```

Do not create files only for aesthetic reasons. Each module must own a coherent state or operation.

---

## 12. `PackedWorkspace`

### Responsibility

Own and resize all reusable buffers.

### Proposed structure

```cpp
class PackedWorkspace {
public:
    PackedWorkspace(
        const PackedWorkspaceRequirements& requirements,
        cudaStream_t stream);

    void ensure_segmented_attention_capacity(
        int rows,
        int chunks);

    void ensure_flattened_prefill_capacity(
        size_t flattened_tokens);

    PackedCoreBuffers& core();
    PackedSamplingBuffers& sampling();
    PackedMetadataBuffers& metadata();
    PackedKvPointerBuffers& kv_pointers();
    PackedMoeBuffers& moe();
    PackedSegmentedAttentionBuffers& segmented();
};
```

### Requirements object

```cpp
struct PackedWorkspaceRequirements {
    size_t maximum_sessions;
    size_t maximum_prefill_tokens;
    int hidden;
    int vocab_size;
    int maximum_attention_projection_width;
    int maximum_dense_intermediate;
    int maximum_moe_intermediate;
    int maximum_experts_per_token;
    int maximum_experts;
    int layers;
    int maximum_pages_per_request;
};
```

### Move into workspace

- `positions`;
- sampled-token buffers;
- generation parameter buffers;
- hidden/residual/norm buffers;
- Q/K/V buffers;
- convolution buffers;
- dense FFN buffers;
- MoE scratch;
- logits scratch;
- sampling scratch;
- host pointer tables;
- device pointer tables;
- page tables;
- span metadata;
- final-row metadata;
- segmented attention buffers;
- flattened seen-pointer buffers.

### Rules

- No `cudaMalloc` or equivalent steady-state allocation inside decode.
- No device allocation inside ragged prefill.
- Growth should be explicit, logged, and testable.
- Capacity should never shrink during executor lifetime unless explicitly reset.
- Workspace sizing must be based on the compiled model program, not one global topology assumption.

### Acceptance criteria

- Allocation tracing shows zero steady-state allocations.
- Workspace requirements are independently unit-testable.
- Decode and prefill pipelines do not own raw scratch buffers.
- Segmented workspace growth occurs only when capacity is exceeded.

---

## 13. `PackedBatchValidator` and Compatibility

### Responsibility

Pure host-side validation.

### Proposed API

```cpp
struct PackedValidationResult {
    const PackedSessionContext* reference;
    PackedExecutionCompatibilityKey key;
    size_t session_count;
    size_t flattened_token_count;
};

class PackedBatchValidator {
public:
    PackedValidationResult validate_decode(
        std::span<const PackedSessionContext> sessions) const;

    PackedValidationResult validate_prefill(
        std::span<const PackedSessionContext> sessions,
        std::span<const int32_t> tokens,
        std::span<const PackedPrefillRow> rows,
        std::span<const PageTableView> page_tables) const;
};
```

### Validate

- non-empty batch;
- capacity;
- null ownership;
- duplicate sessions;
- operation-compatible lifecycle;
- context limit;
- local or paged KV availability;
- page-table length;
- dense ragged offsets;
- non-empty prefill rows;
- same shared weights;
- same compiled packed program;
- same execution-plan fingerprint;
- compatible generation constraints where required.

### Error model

Use structured errors internally:

```cpp
enum class PackedValidationErrorCode {
    EmptyBatch,
    CapacityExceeded,
    DuplicateSession,
    InvalidPhase,
    ContextLimitReached,
    MissingKvStorage,
    IncompatibleProgram,
    IncompatibleExecutionPlan,
    InvalidPageTable,
    InvalidRaggedRows
};
```

Convert to exceptions only at the boundary currently using exceptions.

### Acceptance criteria

- Validator has no CUDA dependency.
- Every failure mode has a focused unit test.
- No pipeline contains repeated batch-validation logic.
- Compatibility is based on immutable keys, not partial option comparisons.

---

## 14. `PackedMetadataStager`

### Responsibility

Convert session state into contiguous host/device metadata required by kernels.

### Split metadata by lifetime

#### Persistent while session layout is unchanged

- logits pointers;
- seen-token pointers;
- RNG pointers;
- sampled destination pointers;
- position destination pointers;
- per-layer KV pointers;
- per-layer convolution state pointers.

#### Per execution step

- positions;
- temperatures;
- repetition penalties;
- top-k;
- top-p;
- page tables;
- segmented attention plan inputs.

#### Per ragged prefill call

- span offsets;
- span counts;
- final rows;
- flattened positions;
- flattened page tables;
- flattened seen pointers.

### Proposed API

```cpp
class PackedMetadataStager {
public:
    void bind_sessions(
        std::span<const PackedSessionContext> sessions);

    PackedAttentionBatchMetadata stage_decode_step(
        std::span<const PackedSessionContext> sessions,
        std::span<const PageTableView> page_tables);

    PackedPrefillMetadata stage_prefill(
        std::span<const PackedSessionContext> sessions,
        std::span<const PackedPrefillRow> rows,
        std::span<const PageTableView> page_tables);
};
```

### Improvements

- Use generation/version IDs instead of comparing only `owner` pointers.
- Detect changes to layer storage even if session identity remains constant.
- Avoid re-copying persistent pointer tables when unchanged.
- Use a compact page-table representation or batched copy where useful.
- Measure copy volume per token.

### Acceptance criteria

- Metadata staging is independently benchmarked.
- Persistent metadata is not copied on every decode if unchanged.
- Prefill does not allocate temporary pointer arrays.
- Session rebinding after cache replacement is correctly detected.

---

## 15. Compiled Packed Layer Program

### Problem

The packed layer loop repeatedly inspects variants and global topology values.

### Target

Compile a backend-specific packed program once.

```cpp
enum class PackedMixerKind {
    Attention,
    ShortConvolution
};

enum class PackedFfnKind {
    Dense,
    Moe
};

struct PackedLayerProgram {
    PackedMixerKind mixer_kind;
    PackedFfnKind ffn_kind;

    LayerCommonBinding common;
    PackedAttentionBinding attention;
    PackedConvolutionBinding convolution;
    PackedDenseFfnBinding dense_ffn;
    PackedMoeBinding moe;
};
```

Only fields relevant to the active kind need to be present, possibly through a compact union or variant outside the hot inner operations.

### Precompute

- layer dimensions;
- pointer bindings;
- KV owner index;
- paged-KV slot;
- projection widths;
- dense intermediate;
- MoE intermediate;
- scale values;
- norm pointers;
- fusion choices;
- residual beta;
- kernel function bindings where stable;
- workspace offsets if one arena is used.

### Layer loop target

```cpp
for (const PackedLayerProgram& layer : program.layers()) {
    execute_operator_norm(layer, workspace, rows);
    execute_mixer(layer, context);
    execute_residual_transition(layer, context);
    execute_ffn(layer, context);
}
```

Architecture identity must not appear in this loop.

### Acceptance criteria

- No repeated `as_attention`, `as_convolution`, `as_dense_ffn`, or `as_moe_ffn` resolution is required in steady-state execution.
- Layer-specific dimensions are immutable and prevalidated.
- The compiled program can be inspected in diagnostics.
- Program creation fails early on unsupported graph combinations.

---

## 16. Operator Executors

### 16.1 Embedding executor

Responsibility:

- invoke the selected weight-layout embedding operation;
- apply embedding scale.

Do not duplicate BF16/INT8/INT4 dispatch if `IWeightLayout` already owns it.

Target:

```cpp
class PackedEmbeddingExecutor {
public:
    void run(
        const PackedEmbeddingBinding& binding,
        std::span<const int32_t> token_ids,
        DeviceMatrixView output,
        cudaStream_t stream);
};
```

### 16.2 Attention executor

Responsibility:

- QKV projection;
- Q/K norm and RoPE;
- KV store;
- local or paged attention;
- BF16 or INT8 KV path;
- segmented or standard path;
- output projection.

Inputs should be explicit:

```cpp
struct PackedAttentionExecutionContext {
    const PackedAttentionBinding& layer;
    const PackedAttentionPlan& plan;
    PackedWorkspace& workspace;
    PackedMetadataDeviceView metadata;
    PhysicalPagedKvCache* paged_kv;
    int rows;
    cudaStream_t stream;
};
```

Do not pass the entire session context unless strictly necessary.

### 16.3 Convolution executor

Responsibility:

- input projection;
- decode or ragged-prefill convolution;
- output projection.

It should not know about sampling or general session lifecycle.

### 16.4 Dense FFN executor

Responsibility:

- FFN norm if not performed by outer loop;
- fused or split gate/up projection;
- activation;
- down projection;
- residual behavior.

Use the layer-specific intermediate width.

### 16.5 MoE executor

Responsibility:

- norm/cast;
- router;
- expert selection;
- residency coordination;
- expert FFN;
- accumulation;
- final residual.

Expert residency is a separate collaborator:

```cpp
class IMoeResidencyCoordinator {
public:
    virtual void ensure_resident(
        const MoeResidencyRequest&) = 0;
};
```

For hot-path avoidance of virtual calls, bind a concrete callback in the compiled executor context.

### Acceptance criteria

- Each executor has focused unit or integration tests.
- Operator executors do not mutate session lifecycle directly.
- Dimensions come from compiled layer bindings.
- No architecture-specific switch exists.
- Weight-layout dispatch is not duplicated.

---

## 17. Decode Pipeline

### Responsibility

Orchestrate one packed decode step.

### Target sequence

```text
validate batch
bind or refresh persistent session metadata
stage step metadata
resolve immutable execution plan
sample previous logits
embed sampled tokens
execute transformer
compute final logits
scatter logits and decode state
synchronize or record completion event
commit host session state
record metrics
```

### Target API

```cpp
class PackedDecodePipeline {
public:
    PackedDecodeResult execute(
        std::span<const PackedSessionContext> sessions,
        std::span<const PageTableView> page_tables);
};
```

### Session commit

Separate GPU launch from host mutation:

```cpp
struct PackedDecodeCompletion {
    std::vector<int32_t> sampled_tokens;
    int position_increment;
    PackedTiming timing;
};
```

Then:

```cpp
commit_decode_completion(sessions, completion);
```

This makes completion logic testable and prepares future asynchronous execution.

### Metrics

Record:

- rows;
- segmented rows;
- local/paged KV path;
- GPU elapsed time;
- host preparation;
- host commit;
- H2D bytes;
- D2H bytes;
- kernel count if instrumented.

### Acceptance criteria

- Decode pipeline source remains small and orchestration-focused.
- No workspace allocation occurs.
- No execution-plan compilation occurs per token.
- Host state is committed only after successful completion.
- Failure behavior does not leave sessions partially advanced.

---

## 18. Ragged Prefill Pipeline

### Responsibility

Orchestrate one flattened ragged prefill pass.

### Target sequence

```text
validate sessions, rows, tokens, and page tables
bind persistent metadata
stage spans and flattened metadata
upload explicit tokens
mark seen tokens
embed all tokens
execute transformer once
gather only final rows requiring logits
run final norm only on gathered rows
run LM head only on gathered rows
scatter final logits and session state
record completion
commit host state
record metrics
```

### Major performance improvement: final-row gather

Current behavior can project vocabulary logits for every flattened token when only final rows are needed.

Introduce:

```cpp
launch_gather_bf16_rows(
    hidden,
    final_rows,
    gathered_hidden,
    finalized_request_count,
    hidden_width,
    stream
);
```

Then execute final norm and LM head only for finalized requests.

This reduces:

- LM-head GEMM rows;
- logits scratch size;
- VRAM;
- memory bandwidth;
- execution time for long ragged prompts.

### Partial prefill

Rows with `finalize == 0` should:

- advance position;
- preserve prefilling phase;
- avoid unnecessary logits projection.

### Asynchrony

Use CUDA events and avoid mandatory stream synchronization unless the public contract requires immediate host-visible results.

### Acceptance criteria

- One transformer pass per ragged batch remains guaranteed.
- LM-head rows equal finalized request count, not flattened token count.
- No temporary device allocation occurs.
- Timing represents actual GPU execution.
- Partial prefill transitions are tested.
- Mixed finalized/non-finalized rows are tested.
- Results match non-ragged reference execution.

---

## 19. Executor Facade

After extraction, the public facade should remain compact.

```cpp
class PackedInferenceExecutor {
public:
    PackedInferenceExecutor(
        std::shared_ptr<const PackedCompiledProgram> program,
        PackedExecutorResources resources,
        PackedExecutorOptions options);

    PackedEligibilityResult eligible(
        const PackedSessionContext& session,
        PackedOperation operation) const;

    PackedDecodeResult decode(
        std::span<const PackedSessionContext> sessions,
        std::span<const PageTableView> page_tables = {});

    PackedPrefillResult prefill(
        std::span<const PackedSessionContext> sessions,
        std::span<const PageTableView> page_tables,
        std::span<const int32_t> tokens,
        std::span<const PackedPrefillRow> rows);

    PackedDecodeMetrics metrics() const;
};
```

Consider renaming from `PackedDecodeExecutor` to `PackedInferenceExecutor`, because it executes decode and prefill.

Provide a compatibility alias or migration period if the class is public.

### Acceptance criteria

- Facade implementation is orchestration only.
- Decode and prefill are separate collaborators.
- Metrics are collected through one focused component.
- Existing callers require minimal migration.

---

# Part VII — CUDA Dispatch and Weight Policies

## 20. Phase 7 — Immutable CUDA Execution Plans

### Problem

Packed decode and prefill compile an execution plan per call.

### Target

Compile once when:

- model options are finalized;
- maximum context is known;
- backend resources are created.

```cpp
class CudaCompiledExecutionPlan {
public:
    uint64_t fingerprint() const;
    const LinearExecutionPolicy& linear() const;
    const AttentionExecutionPolicy& attention() const;
    const KvExecutionPolicy& kv() const;
    const MoeExecutionPolicy& moe() const;
    const FusionPolicy& fusion() const;
};
```

### Plan caching

If sessions may legitimately use different options over shared weights:

```cpp
class CudaExecutionPlanCache {
public:
    std::shared_ptr<const CudaCompiledExecutionPlan> get_or_compile(
        const CudaPlanKey& key);
};
```

The packed batch requires identical plan fingerprints.

### Acceptance criteria

- No plan compile occurs in steady-state decode.
- Plan compilation count is exposed in diagnostics.
- Batch compatibility uses the plan fingerprint.
- Plan objects are immutable and thread-safe.

---

## 21. Phase 8 — Refactor `GemmDispatcher`

### Current concern

`GemmDispatcher` combines:

- CUDA library handles;
- cuBLAS;
- cuBLASLt;
- plan cache;
- heuristics;
- autotuning;
- workspace;
- MMQ scratch;
- quantization-specific execution;
- BF16;
- INT8;
- INT4;
- GGUF layouts;
- GEMV fallback;
- central storage dispatch.

### Target collaborators

```text
CudaLinearExecutor
  ├── Bf16LinearExecutor
  ├── Int8LinearExecutor
  ├── Int4LinearExecutor
  ├── GgufQ4KLinearExecutor
  ├── GgufQ6KLinearExecutor
  └── GemvExecutor

LtPlanCache
LtAutotuner
CudaLinearWorkspace
MmqWorkspace
```

### Hot-path design

Do not necessarily call a virtual interface for every linear operation.

Compile the selected function into the weight or layer binding:

```cpp
using LinearExecutionFn = void (*)(
    const LinearExecutionRequest&,
    CudaLinearRuntime&);

struct CompiledLinearBinding {
    LinearExecutionFn execute;
    LinearKernelKind kind;
    LinearWeight weight;
};
```

### Correct OCP language

A switch over a deliberately closed enum is acceptable, but extending that switch is not itself evidence of OCP compliance.

Document each domain as either:

- **closed and centrally versioned**, or
- **open and registry-driven**.

### Acceptance criteria

- GEMM library lifecycle is separate from quantized-kernel selection.
- New layout execution can be added without editing unrelated code.
- Hot-path dispatch overhead does not regress measurably.
- Plan cache and autotuning can be tested independently.
- Comments accurately describe extension behavior.

---

## 22. Weight Layout Strategy

### Problem

`IWeightLayout` is a useful abstraction, but central factories and packed execution may bypass it.

### Target

A compiled model should receive a concrete layout strategy selected at construction.

Potential API:

```cpp
class IWeightLayout {
public:
    virtual ~IWeightLayout() = default;

    virtual CompiledEmbeddingBinding compile_embedding(
        const LinearWeight&) const = 0;

    virtual CompiledLinearBinding compile_linear(
        const LinearWeight&,
        const LinearCompileContext&) const = 0;
};
```

Virtual calls occur at compile time. Execution uses concrete bindings.

### Acceptance criteria

- Packed embedding uses compiled layout behavior.
- Adding a layout does not require editing packed execution.
- Layout validation occurs before execution.
- Unsupported combinations fail during model compilation.

---

# Part VIII — API and Serving

## 23. Phase 9 — Refactor the C API

### Current responsibility inventory

`src/api/api.cpp` includes:

- C ABI handle definitions;
- exception translation;
- default option initialization;
- struct version validation;
- enum conversion;
- CPU option conversion;
- CUDA option conversion;
- backend construction;
- direct model APIs;
- engine APIs;
- tokenizer APIs;
- lifecycle operations.

### Target layout

```text
src/api/
  handles.hpp
  errors.cpp
  validation.cpp
  option_mapping.cpp
  backend_factory.cpp
  model_api.cpp
  engine_api.cpp
  tokenizer_api.cpp
  diagnostics_api.cpp
```

### Backend factory

Replace hardcoded CPU/CUDA branching with an injected or catalog-driven factory.

```cpp
class IBackendFactory {
public:
    virtual ~IBackendFactory() = default;
    virtual std::string_view id() const = 0;
    virtual BackendCapabilities capabilities() const = 0;
    virtual std::unique_ptr<IBackendModel> create(
        const BackendCreateRequest&) const = 0;
};
```

The C API maps stable enums or strings to registered factories.

### ABI safety

Keep:

- opaque handles;
- versioned structs;
- strict validation;
- no exceptions crossing C boundaries.

Add:

- handle kind tags;
- optional debug generation counters;
- consistent null handling;
- consistent ownership documentation.

### Acceptance criteria

- `api.cpp` no longer exists as a monolith.
- Option mapping has table-driven tests.
- Backend construction is independently testable.
- Adding a backend does not require editing all API entry points.
- ABI compatibility tests pass.

---

## 24. Phase 10 — Refactor Serving Composition

### Current concern

Serving interfaces are segregated, but one concrete service implements:

- request submission;
- polling;
- release;
- scheduler control;
- diagnostics.

CPU and CUDA services duplicate orchestration. The main serving executable also selects:

- checkpoint behavior;
- tokenizer;
- chat profile;
- vision handling;
- mmproj conventions;
- backend;
- routes.

### Target service roles

```cpp
class IRequestService;
class IRequestResultStore;
class ISchedulerController;
class IServiceDiagnostics;
```

Composition:

```cpp
struct ServiceBundle {
    std::shared_ptr<IRequestService> requests;
    std::shared_ptr<IRequestResultStore> results;
    std::shared_ptr<ISchedulerController> scheduler;
    std::shared_ptr<IServiceDiagnostics> diagnostics;
};
```

One object may implement multiple roles, but the bundle should not require it.

### Shared orchestration

Extract backend-neutral request lifecycle:

```text
request validation
tokenization
scheduler submission
result polling
streaming adaptation
release
metrics mapping
```

Backend-specific adapters should provide:

- session creation;
- prefill;
- decode;
- cancellation;
- backend metrics.

### Model capability-driven vision

Do not branch on a chat profile string such as `"gemma4-instruct"`.

Use:

```cpp
if (resolved.capabilities.modalities.vision) {
    vision_provider = runtime.vision_providers().resolve(...);
}
```

Do not encode a fixed `mmproj-BF16.gguf` convention in the generic main function.

### Acceptance criteria

- CPU and CUDA serving code share backend-neutral orchestration.
- Service interfaces do not impose narrower hidden limits in one implementation.
- CUDA output-token limits are represented explicitly as capabilities or validated before generic submission.
- Vision behavior is capability-driven.
- Main composition root is declarative and small.

---

# Part IX — Text, Chat, and Tokenization

## 25. Phase 11 — Refactor Chat Profiles and Templates

### Problems

- concrete chat templates are declared centrally;
- one source file contains templates, JSON serialization, role mapping, tool encoding, parsers, and built-in catalog construction;
- capability flags and raw codec pointers can form inconsistent states;
- some templates reject roles accepted by the nominal common interface.

### Target profile

```cpp
struct ChatProfile {
    std::string id;
    std::shared_ptr<const IChatTemplate> template_engine;
    std::shared_ptr<const IChatToolCallCodec> tool_codec;
    ChatRoleCapabilities roles;
    ToolCapabilities tools;
    MultimodalPromptCapabilities multimodal;
};
```

Invalid states should be rejected on construction.

### Role contract

Instead of assuming every template supports every role:

```cpp
struct ChatRoleCapabilities {
    bool system;
    bool user;
    bool assistant;
    bool tool;
};
```

Validate before formatting.

Alternatively, make formatting return a structured unsupported-role error rather than throwing unexpectedly.

### Source layout

```text
src/text/
  chat_profile_catalog.cpp
  common/json.cpp
  common/tool_serialization.cpp
  profiles/lfm2.cpp
  profiles/granite.cpp
  profiles/gemma4.cpp
  profiles/minicpm5.cpp
```

### Tokenizer providers

Resolve tokenizer support through a catalog or provider abstraction. Serving should not need concrete repository casts to construct tokenizers.

### Acceptance criteria

- A chat profile can be registered through `RuntimeBuilder`.
- Capability flags and codec lifetime are coherent.
- Unsupported roles are validated explicitly.
- Adding a profile does not require editing one central template file.
- Serving constructs tokenizers through providers, not checkpoint-type casts.

---

# Part X — Graph Extensibility

## 26. Phase 12 — Reassess Closed Graph Variants

### Current state

The graph currently uses closed variants such as:

```cpp
variant<AttentionSpec, ShortConvolutionSpec>
variant<DenseFeedForwardSpec, MixtureOfExpertsSpec>
```

This is not automatically wrong.

Advantages:

- compact representation;
- exhaustive compiler handling;
- no virtual dispatch;
- easy validation;
- explicit supported operator universe.

Disadvantages:

- every new mixer or FFN family requires central edits;
- backend compilers must be updated;
- visitors and variants grow;
- external operator plugins are difficult;
- architectures such as Mamba, DeltaNet, RWKV, or future hybrid operators may stress the design.

### Decision gate

Do not immediately replace variants with inheritance.

First classify the intended extension model.

#### Option A — Closed internal operator universe

Keep variants if Celeg intentionally supports a centrally controlled set of operator families.

Requirements:

- document the domain as closed;
- central changes are expected;
- exhaustive compiler errors are valuable;
- extension tests cover every backend;
- do not claim plugin-level OCP.

#### Option B — Open operator compiler registry

Introduce stable operator IDs and compiler registries:

```cpp
struct OperatorNode {
    OperatorTypeId type;
    OperatorAttributes attributes;
};

class IOperatorCompiler {
public:
    virtual CompiledOperator compile(
        const OperatorNode&,
        const BackendCompileContext&) const = 0;
};
```

Execution still uses backend-specific compiled bindings.

### Recommended approach

Use a two-level model:

1. stable neutral operator categories for common families;
2. backend-specific compiled operator bindings.

Keep the neutral graph explicit, but avoid architecture switches in backends.

### Acceptance criteria

- The project documents whether graph operator types are open or closed.
- Adding one test mixer has a measured change surface.
- Unsupported backend/operator combinations fail during compilation.
- Hot-path execution remains concretely bound.

---

# Part XI — Contracts and SOLID Completion

## 27. Phase 13 — Strengthen LSP and Capability Contracts

### 27.1 Serving limits

If CUDA requires `max_output_tokens <= INT_MAX`, do not hide that narrower precondition inside an implementation of a broader interface.

Options:

- use a common validated type;
- publish backend limits;
- reject at service construction;
- make the interface use a type whose range matches all implementations.

Example:

```cpp
struct ServiceLimits {
    uint64_t maximum_output_tokens;
    uint64_t maximum_context_tokens;
    uint64_t maximum_concurrent_requests;
};
```

### 27.2 Chat-template role support

Do not expose an interface suggesting universal role support when Granite or another profile rejects tool messages.

Use capabilities or a result type.

### 27.3 Optional repository operations

Continue the existing good pattern of separate capability interfaces.

Avoid adding optional methods returning “unsupported” to a broad base interface when capability interfaces can model them safely.

### 27.4 Raw pointer cleanup

Audit:

- tensor naming policies;
- chat codecs;
- service bundle role pointers;
- session context pointers;
- architecture-owned statics;
- model provenance strings.

For each pointer document:

- owner;
- lifetime;
- thread safety;
- mutation;
- invalidation condition.

Prefer stable shared immutable ownership where overhead is not in the hot loop.

### Acceptance criteria

- No implementation imposes undocumented narrower preconditions.
- Optional features are represented through explicit capabilities.
- Raw non-owning pointers have verified lifetime or are replaced.
- Contract tests run across CPU and CUDA implementations.

---

# Part XII — Automation and Governance

## 28. Phase 14 — Expand Architecture Boundary Checks

### Existing good foundation

The project already has structural checks for:

- CUDA leakage into neutral headers;
- architecture names in backend code;
- source-format types in backend code;
- architecture-specific types in generic runtime;
- build graph boundaries.

### Add rules

#### Backend checkpoint isolation

Reject in backend directories:

```text
GgufRepository
SafeTensorRepository
dynamic_cast<...Repository>
checkpoint/gguf
checkpoint/safetensors
```

#### Architecture identity leakage

Reject in backend runtime code:

```text
architecture_id ==
model_type ==
chat_profile_id ==
```

Allow identity only in diagnostics or explicit composition modules.

#### Chat-profile string branching

Reject profile-string decisions outside:

```text
text profile registration
composition roots
diagnostics
```

#### Hot-path plan compilation

Reject calls to:

```text
CudaExecutionPlan::compile
```

inside known execution paths such as decode, prefill, or layer loops.

#### Steady-state allocation lint

Flag obvious:

- `DeviceBuffer` construction inside decode/prefill;
- `cudaMalloc` inside execution paths;
- temporary vectors in known hot paths.

This will not replace profiling, but it catches regressions.

#### Concrete factory switches

Track central switches and require annotation:

```cpp
// CELEG_CLOSED_DOMAIN: WeightMode
```

This forces maintainers to declare that the switch is intentionally closed.

### Extension tests

Add compile-time or runtime tests proving:

1. custom architecture registration;
2. custom checkpoint registration;
3. custom chat profile registration;
4. custom tokenizer provider registration;
5. fake backend registration if public;
6. no edits to built-in central files are required.

### Acceptance criteria

- CI rejects concrete checkpoint dependencies in backends.
- CI rejects architecture/profile strings in execution code.
- CI detects execution-plan compilation in hot paths.
- Extension tests use only public registration APIs.
- Boundary exceptions are explicit and reviewed.

---

# Part XIII — Performance Validation

## 29. Phase 15 — Benchmark and Optimize After Refactoring

### Required benchmark comparisons

For every major phase compare against Phase 0.

#### Decode

- batch 1;
- batch 2;
- batch 4;
- batch 8;
- batch 16;
- maximum supported packed batch.

Measure:

- aggregate tokens/s;
- per-request latency;
- kernel launches/token;
- H2D bytes/token;
- CPU overhead/token;
- GPU utilization;
- GEMM occupancy;
- attention occupancy;
- synchronization count.

#### Prefill

Use ragged prompts such as:

```text
[16, 32, 64, 128]
[4, 512, 32, 2048]
[1024, 1024, 1024, 1024]
mixed finalize flags
```

Measure:

- flattened tokens/s;
- finalized rows;
- LM-head rows;
- logits scratch VRAM;
- host allocation count;
- device allocation count;
- page-table copy volume.

#### Model shapes

Test:

- normal hidden/query width;
- non-equal hidden/query width;
- variable FFN intermediate;
- dense;
- MoE;
- attention-only;
- convolution/attention hybrid;
- shared KV;
- paged and local KV.

### Regression policy

Suggested initial thresholds:

| Metric | Allowed regression |
|---|---:|
| Correctness | zero unexplained regressions |
| Decode tokens/s | no more than 2% |
| Prefill tokens/s | no more than 2% |
| p95 latency | no more than 3% |
| VRAM | no more than 2%, unless explicitly traded for speed |
| Host allocation count | must improve or remain zero in steady state |
| Device allocation count | zero in steady-state decode and prefill |

Thresholds should be hardware-specific and statistically evaluated.

### Expected improvements

Potential gains from the plan:

- no per-token plan compilation;
- no per-prefill device buffer allocation;
- smaller prefill LM-head GEMM;
- smaller logits scratch;
- fewer repeated variant inspections;
- fewer persistent metadata copies;
- more predictable workspace;
- better profiling visibility;
- easier specialized kernel binding.

### Acceptance criteria

- Performance reports are attached to each relevant change.
- No architectural refactor is merged with unknown hot-path impact.
- CUDA event metrics agree with profiler timelines.
- Improvements and regressions are documented by model and hardware.

---

# Part XIV — Documentation and Migration

## 30. Phase 16 — Documentation

Create or update:

```text
docs/ARCHITECTURE.md
docs/ARCHITECTURE_RULES.md
docs/EXTENDING_CELEG.md
docs/BACKEND_COMPILATION.md
docs/PACKED_EXECUTION.md
docs/CHECKPOINT_FORMATS.md
docs/CHAT_PROFILES.md
docs/PERFORMANCE_VALIDATION.md
docs/MIGRATION_GUIDE.md
```

### `EXTENDING_CELEG.md`

Include complete examples for:

- registering an architecture;
- registering a checkpoint format;
- registering a chat profile;
- registering a tokenizer provider;
- compiling a new operator;
- adding a CUDA weight layout;
- adding a backend.

### `PACKED_EXECUTION.md`

Document:

- session compatibility;
- workspace lifetime;
- packed program compilation;
- metadata staging;
- local vs paged KV;
- BF16 vs INT8 KV;
- segmented attention;
- ragged prefill;
- final-row gather;
- metrics semantics;
- asynchronous behavior;
- failure and rollback behavior.

### Acceptance criteria

- A contributor can add a test architecture using documentation alone.
- Packed execution invariants are explicit.
- Closed domains are labeled as closed.
- No documentation describes “add another switch case” as Open/Closed compliance.

---

# Part XV — Recommended Commit Sequence

## 31. Incremental Implementation Order

The following order minimizes risk.

### Milestone A — Safety net

1. Add packed/non-packed parity tests.
2. Add synthetic nontraditional-shape fixtures.
3. Add allocation instrumentation.
4. Add CUDA event timing.
5. Store benchmark baseline.

### Milestone B — Packed correctness

1. Fix layer-specific attention widths.
2. Fix layer-specific dense FFN widths.
3. Add full compatibility key.
4. Centralize lifecycle validation.
5. Fix prefill timing.

### Milestone C — Packed allocation and LM-head improvements

1. Move flat seen-pointer buffers into persistent workspace.
2. Remove per-call device allocation.
3. Add final-row gather.
4. Project logits only for finalized rows.
5. Benchmark memory and throughput.

### Milestone D — Composition inversion

1. Introduce `RuntimeBuilder`.
2. Wrap existing built-ins.
3. Inject catalogs into bootstrap.
4. Add custom-extension tests.
5. Deprecate global catalog access.

### Milestone E — Checkpoint/backend boundary

1. Add repository capability interfaces.
2. Fully resolve weight-plan names.
3. Remove concrete repository casts from CPU.
4. Remove concrete repository knowledge from CUDA if present.
5. Strengthen boundary checker.

### Milestone F — Packed workspace extraction

1. Create requirements object.
2. Move core buffers.
3. Move metadata buffers.
4. Move MoE scratch.
5. Move segmented scratch.
6. Verify zero performance regression.

### Milestone G — Packed staging and validation extraction

1. Extract validator.
2. Extract compatibility key.
3. Extract persistent metadata binding.
4. Extract per-step staging.
5. Extract ragged staging.

### Milestone H — Packed compiled program

1. Compile layer-specific bindings.
2. Prevalidate all shapes.
3. Prebind weight-layout execution.
4. Remove repeated variant resolution.
5. Cache immutable execution plan.

### Milestone I — Packed pipeline split

1. Extract attention executor.
2. Extract convolution executor.
3. Extract dense FFN executor.
4. Extract MoE executor.
5. Extract transformer executor.
6. Extract decode pipeline.
7. Extract prefill pipeline.
8. Rename facade if desired.

### Milestone J — Broader SRP cleanup

1. Split architecture families.
2. Split C API.
3. Split serving composition.
4. Split chat profiles.
5. Split GEMM dispatcher.

### Milestone K — Contract and extension completion

1. Publish backend limits.
2. Fix role capability contracts.
3. Remove unsafe raw policy pointers.
4. Decide open vs closed graph operator model.
5. Complete extension documentation.

---

# Part XVI — Detailed Task Backlog

## 32. P0 Backlog

### P0.1 Packed shape correctness

- [ ] Add synthetic attention model where `query_width != hidden`.
- [ ] Add synthetic model with per-layer FFN intermediate widths.
- [ ] Replace output-projection global width assumption.
- [ ] Replace dense-FFN global intermediate assumption.
- [ ] Validate workspace maximums against compiled layer program.
- [ ] Add shape mismatch diagnostics with layer index and semantic name.

### P0.2 Compatibility correctness

- [ ] Define `CudaPlanKey`.
- [ ] Define stable hash/fingerprint.
- [ ] Include all execution-relevant options.
- [ ] Include compiled program identity.
- [ ] Replace `options_compatible`.
- [ ] Test incompatible GEMM backend.
- [ ] Test incompatible attention chunk size.
- [ ] Test incompatible fusion settings.
- [ ] Test incompatible MoE offload behavior.

### P0.3 Metrics correctness

- [ ] Add reusable CUDA event wrapper if absent.
- [ ] Measure prefill GPU time.
- [ ] Separate host prepare and commit time.
- [ ] Document metric definitions.
- [ ] Add profiler correlation test.

### P0.4 Allocation correctness

- [ ] Instrument `DeviceBuffer` allocation.
- [ ] Move `d_flat_seen` into persistent workspace.
- [ ] Move host flattened pointer arrays into persistent workspace.
- [ ] Assert no steady-state allocation in tests.

---

## 33. P1 Backlog

### P1.1 Runtime composition

- [ ] Implement `RuntimeBuilder`.
- [ ] Implement immutable `RuntimeContext`.
- [ ] Register built-ins through functions.
- [ ] Inject runtime into bootstrap.
- [ ] Add minimal test runtime.
- [ ] Add duplicate-ID validation.
- [ ] Document thread safety.

### P1.2 Weight-plan resolution

- [ ] Define `ResolvedWeightRequest`.
- [ ] Resolve tensor names in architecture layer.
- [ ] Attach transforms and expected shapes.
- [ ] Remove backend naming-policy dependency.
- [ ] Remove concrete checkpoint repository dependencies.
- [ ] Add fake repository test.

### P1.3 Packed workspace

- [ ] Create `PackedWorkspaceRequirements`.
- [ ] Extract core buffers.
- [ ] Extract metadata buffers.
- [ ] Extract KV pointer buffers.
- [ ] Extract sampling buffers.
- [ ] Extract MoE buffers.
- [ ] Extract segmented attention buffers.
- [ ] Add capacity-growth tests.

### P1.4 Packed validation and staging

- [ ] Extract pure validator.
- [ ] Add structured error codes.
- [ ] Extract session binding.
- [ ] Track session storage generations.
- [ ] Extract decode-step staging.
- [ ] Extract ragged prefill staging.
- [ ] Benchmark staging overhead.

### P1.5 Packed program and executors

- [ ] Compile attention bindings.
- [ ] Compile convolution bindings.
- [ ] Compile dense FFN bindings.
- [ ] Compile MoE bindings.
- [ ] Bind embedding strategy.
- [ ] Bind linear strategies.
- [ ] Extract transformer loop.
- [ ] Extract final projection.
- [ ] Split decode and prefill pipelines.

### P1.6 Prefill optimization

- [ ] Add final-row gather kernel.
- [ ] Allocate gathered-hidden scratch by maximum finalized sessions.
- [ ] Run final norm on gathered rows.
- [ ] Run LM head on gathered rows.
- [ ] Reduce logits scratch capacity.
- [ ] Test mixed finalize flags.
- [ ] Benchmark long ragged prompts.

### P1.7 API and serving

- [ ] Split C API modules.
- [ ] Add backend factory catalog.
- [ ] Extract shared serving lifecycle.
- [ ] Separate scheduler controller.
- [ ] Separate diagnostics role.
- [ ] Make vision capability-driven.
- [ ] Remove tokenizer repository casts.

---

## 34. P2 Backlog

### P2.1 Model contracts

- [ ] Remove duplicated topology/definition fields.
- [ ] Split provenance from execution.
- [ ] Define coherent token policy.
- [ ] Define coherent numerical policy.
- [ ] Make per-layer dimensions authoritative.
- [ ] Remove unresolved policy pointers from compiled program.

### P2.2 Architecture modules

- [ ] Split LFM2 implementation.
- [ ] Split Granite implementation.
- [ ] Split Gemma4 implementation.
- [ ] Split MiniCPM5 implementation.
- [ ] Add unit tests per stage.
- [ ] Add family registration functions.

### P2.3 Chat and tokenizer

- [ ] Define owned `ChatProfile`.
- [ ] Replace raw codec pointers.
- [ ] Split profile source files.
- [ ] Validate role support.
- [ ] Add tokenizer providers.
- [ ] Register profiles through runtime builder.

### P2.4 GEMM and layouts

- [ ] Separate handle ownership.
- [ ] Separate Lt plan cache.
- [ ] Separate autotuner.
- [ ] Separate quantized executors.
- [ ] Compile linear function bindings.
- [ ] Remove packed-path layout duplication.

### P2.5 Boundary automation

- [ ] Reject concrete repository types in backends.
- [ ] Reject profile-string branching in execution.
- [ ] Reject architecture branching in backends.
- [ ] Detect plan compilation in hot paths.
- [ ] Detect obvious hot-path allocation.
- [ ] Add public extension tests.

---

# Part XVII — Test Plan

## 35. Unit Tests

### Runtime composition

- custom architecture registration;
- custom checkpoint format registration;
- duplicate ID;
- catalog freeze;
- context lifetime;
- concurrent read access.

### Architecture resolution

- probe scoring;
- ambiguous architecture;
- missing metadata;
- malformed metadata;
- shape validation;
- token extraction;
- graph construction;
- weight-plan names.

### Packed validator

- empty batch;
- duplicate session;
- invalid phase;
- context exhausted;
- missing local KV;
- missing paged KV;
- incompatible weights;
- incompatible plan;
- invalid page table;
- ragged gap;
- ragged overlap;
- zero-token row;
- capacity overflow.

### Workspace

- initial sizing;
- segmented growth;
- flattened prefill growth;
- no shrink;
- bounds;
- maximum model dimensions.

### Layer program

- attention dimensions;
- shared KV owner;
- convolution;
- variable FFN;
- MoE;
- unsupported operator;
- invalid weight shapes.

---

## 36. Integration Tests

### CPU and CUDA backend neutrality

- load from fake repository;
- load from GGUF;
- load from SafeTensors;
- compare resolved model;
- ensure backend source format independence.

### Packed execution

- batch 1 parity;
- batch N parity;
- local KV;
- paged KV;
- INT8 KV;
- segmented attention;
- shared KV;
- short convolution;
- dense FFN;
- MoE;
- variable FFN width;
- non-equal query width;
- final logit softcap.

### Ragged prefill

- one request;
- multiple equal lengths;
- highly uneven lengths;
- partial rows;
- mixed finalize;
- context boundary;
- page boundary;
- long prompts;
- gathered logits parity.

### Serving

- CPU request lifecycle;
- CUDA request lifecycle;
- cancellation;
- output limit validation;
- diagnostics;
- tool calls;
- unsupported role;
- vision capability selection.

---

## 37. Performance Tests

Add automated microbenchmarks for:

```text
packed validation
persistent metadata bind
step metadata stage
page table flatten
embedding
attention
convolution
dense FFN
MoE
final-row gather
LM head
session commit
```

Add end-to-end benchmarks for:

- single decode;
- packed decode;
- standard prefill;
- ragged prefill;
- serving scheduler.

Use warmup and statistical repetition.

---

# Part XVIII — Risks and Mitigations

## 38. Risk: Abstraction Overhead

### Risk

Refactoring into collaborators may introduce:

- virtual dispatch;
- pointer chasing;
- reduced inlining;
- larger compile-time interfaces;
- additional synchronization.

### Mitigation

- virtual calls only during construction/compilation;
- immutable compiled bindings;
- direct function pointers in hot paths;
- contiguous program arrays;
- benchmark every extraction;
- inspect generated SASS where critical.

---

## 39. Risk: Hidden Behavior Changes

### Risk

Large monolithic code often contains implicit ordering and synchronization dependencies.

### Mitigation

- characterization tests first;
- one responsibility extraction per commit;
- preserve launch order initially;
- add CUDA event and stream assertions;
- use Nsight Systems comparison;
- do not combine structural and algorithmic changes unless necessary.

---

## 40. Risk: Pointer Lifetime and Session Mutation

### Risk

`PackedSessionContext` exposes many borrowed pointers. Extraction may accidentally lengthen their lifetime or cache invalid state.

### Mitigation

- introduce storage generation IDs;
- document invalidation;
- bind sessions per layout generation;
- avoid caching raw pointers across model storage replacement without validation;
- consider stable session-state objects owned by `shared_ptr`.

---

## 41. Risk: Workspace Memory Growth

### Risk

A generalized workspace based on maxima may consume excessive VRAM.

### Mitigation

- separate decode and prefill capacity classes;
- separate dense and MoE scratch where lifetimes do not overlap;
- use arena aliasing for non-overlapping buffers;
- calculate lifetime intervals;
- expose workspace estimates before allocation;
- support configurable capacity;
- retain lazy growth for segmented attention and rare paths.

---

## 42. Risk: Extension API Instability

### Risk

Publishing registries too early may freeze weak interfaces.

### Mitigation

- mark extension APIs experimental initially;
- provide semantic versioning rules;
- keep neutral metadata views narrow;
- prefer value objects;
- add real external test plugins or examples before stabilization.

---

# Part XIX — Definition of Done

## 43. Architectural Definition of Done

The refactoring is complete when:

- backends do not depend on concrete checkpoint formats;
- runtime composition is injectable;
- architectures can be registered without editing central factories;
- chat profiles and tokenizers can be registered coherently;
- resolved-model fields have one authoritative owner;
- compiled programs contain only execution-relevant state;
- optional capabilities are explicit;
- raw pointer lifetimes are controlled;
- graph extension policy is documented;
- architecture boundary CI enforces the intended direction.

---

## 44. Packed Execution Definition of Done

`packed_execution.cu` is considered successfully refactored when:

- no single implementation object owns every packed subsystem;
- workspace is a dedicated component;
- validation is pure host-side logic;
- metadata staging is isolated;
- layer dimensions are compiled per layer;
- execution plans are immutable and reused;
- embedding dispatch uses the weight-layout strategy;
- attention, convolution, dense FFN, and MoE execution are separated;
- decode and prefill are separate pipelines;
- prefill projects logits only for required final rows;
- no steady-state host or device allocation occurs;
- metrics measure actual GPU execution;
- batch compatibility is complete;
- synthetic nontraditional shapes pass;
- performance is equal or better within defined thresholds.

---

## 45. Final Recommended Priority

The highest-value implementation order is:

1. **Create correctness and performance baselines.**
2. **Fix packed shape assumptions and compatibility.**
3. **Fix prefill timing and allocations.**
4. **Project logits only for finalized ragged rows.**
5. **Introduce injectable runtime composition.**
6. **Remove concrete checkpoint dependencies from backends.**
7. **Extract packed workspace, validation, and metadata staging.**
8. **Compile immutable packed layer and execution plans.**
9. **Split packed operator executors and pipelines.**
10. **Split architecture modules, API, serving, chat, and GEMM responsibilities.**
11. **Strengthen automated architecture enforcement.**
12. **Document extension points and performance contracts.**

The first objective should not be to make every source file short. The first objective should be to make each subsystem have one clear reason to change while preserving Celeg’s performance-oriented design.

The critical architectural pattern is:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

That pattern should guide the entire Celeg runtime, and especially the CUDA packed decode and ragged prefill paths.
