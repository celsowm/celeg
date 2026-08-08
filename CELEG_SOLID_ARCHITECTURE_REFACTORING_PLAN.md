# CELEG SOLID Architecture Refactoring Plan

Status: active execution plan  
Baseline reviewed: `master` at `2d25c56ea052480ab7fcac818a6ba5e816fc11fa`  
Date: 2026-08-07  
Scope: SOLID architecture, semantic model representation, CPU execution, runtime composition, tokenizer ownership, quantized weight extensibility, C API cleanup, and architectural enforcement

This document supersedes `CELEG_SOLID_EXECUTION_AND_TOKENIZER_REFACTORING_PLAN.md` as the primary architecture refactoring plan. The older document remains useful as historical context, but several of its proposed extractions are already implemented and the current remaining debt is now more precise.

---

## 1. Non-negotiable project directives

### 1.1 No backward-compatibility commitment

CELEG currently has **no source, ABI, API, configuration, file-layout, or internal compatibility commitment** unless an explicit future stability milestone says otherwise.

Refactors must replace obsolete interfaces rather than version, alias, deprecate, or retain compatibility shims.

Do not introduce or preserve patterns such as:

- `foo_v2`, `foo_v3`;
- `legacyFoo`;
- `deprecatedFoo`;
- `compatFoo`;
- duplicate old/new API entry points;
- compatibility typedefs;
- transitional aliases;
- wrappers whose only purpose is preserving an obsolete signature;
- alternate code paths solely for backward compatibility.

If a design is replaced, migrate repository callers and delete the superseded design in the same refactoring series.

Compatibility work must only be added after an explicit project decision that names the exact compatibility surface and stability level being protected.

### 1.2 Performance is an architectural constraint

SOLID refactoring must not turn decode/prefill hot paths into general-purpose dynamic object graphs.

Prefer:

- load-time compilation;
- pre-resolved function pointers;
- compact tagged execution records;
- immutable execution programs;
- contiguous reusable scratch storage;
- zero-allocation steady-state execution;
- backend-local specialization;
- static or load-time dispatch where possible.

Avoid:

- architecture probing during decode;
- checkpoint-format probing during decode;
- repeated `dynamic_cast` in hot loops;
- per-layer heap allocation during inference;
- virtual dispatch for every primitive operation when a compiled function target can be selected once.

### 1.3 Architecture identity must not leak into generic execution

Backends execute resolved semantics. They must not ask whether a model is LFM, Granite, Gemma, Qwen, Nemotron, Nanbeige, or any future family.

Architecture identity belongs to checkpoint interpretation and provenance only.

### 1.4 Checkpoint format is not model semantics

Safetensors, GGUF, packed INT8, GPTQ, AWQ, future FP8/MXFP formats, and similar storage representations belong to repository/binding/weight-codec boundaries.

A model graph must describe mathematical semantics, not the physical storage format used by a checkpoint.

---

## 2. Current SOLID assessment

Current approximate assessment at the reviewed baseline:

| Principle | Score | Current state |
| --- | ---: | --- |
| Single Responsibility | 7.5/10 | Stronger module separation now exists, but semantic state and CPU orchestration remain duplicated or oversized. |
| Open/Closed | 7.2/10 | Architecture/backend catalogs are good; family composition, boundary checks, and weight representations still require central edits. |
| Liskov Substitution | 8.8/10 | Capability interfaces are substantially better than optional methods on oversized base interfaces. |
| Interface Segregation | 9.2/10 | Checkpoint repository capabilities are a strong design and should be preserved. |
| Dependency Inversion | 8.0/10 | Generic runtime depends on abstractions, but some CPU operators still depend on the concrete `CpuCompiledModel` world. |

Target after this plan: approximately **9.2-9.5/10**, without sacrificing runtime performance for abstract purity.

A literal 10/10 is not a useful target for a low-level inference runtime. The goal is maintainable extensibility with explicit ownership and compiled hot paths.

---

## 3. Existing architecture strengths to preserve

The following are architectural assets and should not regress.

### 3.1 `IArchitecture` and backend-neutral resolution

An architecture probes checkpoint metadata and resolves a backend-neutral `ResolvedModel`.

Backends must continue consuming resolved semantics rather than model-family identifiers.

### 3.2 Capability-oriented checkpoint interfaces

Preserve the separation among:

- `IWeightRepository`;
- `ILocatableTensorRepository`;
- `IRandomAccessTensorReader`;
- `INativeBlockStorageRepository`;
- `ITokenizerDataRepository`.

Do not collapse them into a larger interface with unsupported/default methods.

### 3.3 Provider catalogs and runtime builder

Preserve the extension direction represented by:

- `IArchitecture`;
- `IRuntimeModule`;
- `IBackendFactory`;
- `IBackendOptionsDecoder`;
- `ITokenizerProvider`;
- `IVisionProviderFactory`;
- provider catalogs;
- frozen `RuntimeContext` ownership.

### 3.4 Compiled semantic program

`CompiledModelProgram` is the correct direction: resolve semantics before execution and give backends immutable execution information.

The plan below strengthens this rather than replacing it.

### 3.5 CPU/CUDA implementation separation

Backend-specific storage, kernels, caches, options, and execution resources must remain backend-local.

---

# Phase 0 - Lock architectural invariants with tests and static checks

Priority: critical  
Risk: low  
Purpose: prevent structural regressions while the remaining refactor proceeds.

## 0.1 Make architecture-boundary checks structural, not family-name based

The current `scripts/check_architecture_boundaries.py` still contains a hard-coded regex naming known families. That approach is itself not Open/Closed and already creates a blind spot when a new family such as Nanbeige is added without updating the regex.

Replace family-name enumeration with dependency rules.

Examples of rules to enforce:

```text
src/backend/**            must not include celeg/models/**
src/runtime/**            must not include celeg/models/**
src/checkpoint/**         must not include celeg/models/**
include/celeg/model/**     must not include celeg/models/**
include/celeg/runtime/**   must not include celeg/models/**
include/celeg/checkpoint/** must not include celeg/models/**
```

Also continue rejecting architecture dispatch tokens inside backend code, including:

- `architecture_id` used for execution decisions;
- `architecture_kind`;
- `model_type` used for execution decisions.

The checker must scale to the 100th model family without editing a list of model names.

## 0.2 Add a no-compatibility-debt checker

Add a static policy check that flags newly introduced compatibility naming in production interfaces unless explicitly allow-listed.

At minimum scan for suspicious public or runtime symbols containing:

```text
_v2
_v3
legacy
deprecated
compat
old_
new_
```

The purpose is not to ban those English words everywhere; it is to prevent accidental API version stacking and migration shims.

Allow-list legitimate format versions only where versioning is intrinsic to an external standard or storage format.

## 0.3 Preserve backend-extension regression tests

Keep and expand the current fake-backend test proving that a new `IBackendFactory` can be registered and selected without modifying the backend catalog implementation.

Add a similar test for a backend exposing `IBackendOptionsDecoder`.

## 0.4 Add family-extension regression coverage

Create a minimal test-only architecture family module that contributes:

- architecture;
- chat profile;
- tokenizer behavior if required;
- optional vision provider.

The test should demonstrate that a coherent family can be added through its module without editing generic runtime code.

## 0.5 Acceptance gate

- boundary checker contains no list of built-in family names;
- adding a fake architecture family requires no generic-runtime edits;
- adding a fake backend requires no generic-backend registry edits;
- new compatibility/version suffixes in CELEG-owned APIs fail static policy checks unless explicitly justified.

---

# Phase 1 - Make `ModelGraph` the canonical semantic source of truth

Priority: critical  
Risk: high if left unresolved  
Purpose: eliminate duplicated model semantics and semantic drift.

This is the most important remaining architecture change.

Today CELEG still carries overlapping semantic representations across:

- `RuntimeTopology`;
- `ModelGraph`;
- `CompiledModelProgram`.

The problem is not that multiple data structures exist. The problem is that multiple structures currently carry independently maintained descriptions of the same semantics.

## 1.1 Target data flow

Target:

```text
Checkpoint
    |
    v
Architecture resolver
    |
    v
ModelGraph + model policies + weight plan
    |
    +-----------------------+
    |                       |
    v                       v
Derived runtime shape   CompiledModelProgram
    |                       |
    +-----------+-----------+
                |
                v
        backend compilation
```

`ModelGraph` must be the semantic authority.

Derived shape/workspace information may exist, but it must be calculated from the graph and immutable policies rather than populated separately by each architecture.

## 1.2 Stop building the graph from `RuntimeTopology`

The current resolution order is effectively:

```text
checkpoint -> RuntimeTopology -> graph
```

Invert that responsibility.

Architecture modules should resolve semantic layer specifications directly.

For example, instead of setting:

- `mixer_kinds`;
- `attention_layouts`;
- `feed_forward_kinds`;
- `feed_forward_intermediates`;
- `gated_delta_net_layouts`;
- `mamba2_layouts`;

and then translating them into `LayerSpec`, construct `LayerSpec` directly.

## 1.3 Introduce a derived runtime shape/view

Replace semantic fields in `RuntimeTopology` with a smaller derived object, for example:

```cpp
struct RuntimeShape {
    int hidden = 0;
    int vocab_size = 0;
    int num_layers = 0;
    int max_position_embeddings = 0;

    int max_attention_projection_width = 0;
    int max_attention_query_width = 0;
    int max_feed_forward_intermediate = 0;
    int max_recurrent_scratch = 0;

    TokenPolicy token_policy;
    NumericalPolicy numerical_policy;
};
```

Exact naming is flexible. The rule is not.

A derived runtime object may contain:

- dimensions;
- maxima needed for allocation;
- counts;
- lookup tables derived from the graph;
- token policy;
- global numerical policy where truly global.

It must not independently define the model's layer semantics.

## 1.4 Move `norm_after_layers` into canonical semantics

`norm_after_layers` currently exists as topology/program information. For architectures that intentionally apply final normalization at intermediate boundaries, this is execution semantics and should have one canonical owner.

Possible representations:

- explicit boundary operation in `ModelGraph`;
- per-layer `post_layer_normalization` semantic property;
- a graph-level ordered boundary list owned by `ModelGraph`.

`CompiledModelProgram` should copy/compile that canonical semantic declaration.

## 1.5 Physical checkpoint layer mapping is binding, not execution semantics

`checkpoint_layer_for_layer` describes how executable logical layers bind to checkpoint storage.

Move this responsibility toward weight/binding planning rather than generic runtime topology.

A model may execute logical layers that reuse a physical source block. Execution should only see the resolved logical layer's weights/program; it should not care why two logical layers came from the same physical checkpoint block.

## 1.6 Build `CompiledModelProgram` only from canonical semantic state

After the change, `build_model_program()` must not need to combine semantics from both `model.graph` and semantic fields hidden in topology.

It should compile:

```text
ModelGraph
ModelPolicy
WeightPlan
        |
        v
CompiledModelProgram
```

## 1.7 Validation strategy

`ResolvedModel::validate()` should verify cross-object invariants where the objects serve genuinely different responsibilities:

- graph dimensions vs global dimensions;
- weight plan references valid logical layers;
- capabilities match semantic requirements;
- binding plan covers required roles.

It should not be responsible for proving two duplicated semantic models happen to agree.

## 1.8 Acceptance gate

- architecture modules directly produce canonical graph semantics;
- generic graph builders no longer reconstruct semantics from topology arrays;
- `RuntimeTopology` is removed or reduced to a derived allocation/execution shape;
- mixer/FFN semantics exist in one canonical place;
- `CompiledModelProgram` derives semantics from the canonical graph;
- adding a new mixer property does not require updating multiple parallel semantic descriptions.

---

# Phase 2 - Compile CPU layers instead of interpreting them twice

Priority: critical  
Risk: moderate after semantic tests  
Purpose: remove the remaining token/chunk semantic duplication.

CELEG already extracted significant operator logic from `model_forward.cpp` into:

- attention;
- feed-forward;
- MoE;
- recurrent operators.

That is good progress. The remaining problem is that `forward_token()` and `forward_chunk()` still implement two large layer orchestration paths.

## 2.1 Introduce `CpuCompiledLayer`

Compile each semantic layer once into a CPU-specific executable record.

Possible direction:

```cpp
struct CpuCompiledLayer {
    CpuTokenLayerFn token;
    CpuChunkLayerFn chunk;
    CpuLayerWeights weights;
    CpuLayerExecutionSpec execution;
};
```

Alternative forms are acceptable if they preserve the same ownership.

The key requirement is that dispatch is resolved once during CPU model compilation/loading rather than reconstructed during every forward pass.

## 2.2 One semantic layer pipeline

The layer compiler should encode the semantic pipeline:

```text
pre-operator norm
mixer
mixer scaling
post-mixer norm if configured
residual
FFN pre-norm if configured
FFN / MoE
FFN scaling
post-FFN norm if configured
residual
per-layer input if configured
post-layer boundary operation if configured
```

Token and chunk implementations may use different kernels, but they must be two execution strategies for the same compiled layer semantics.

## 2.3 Remove semantic condition reconstruction from `forward_token()`

`forward_token()` should primarily:

- prepare token/raw embedding input;
- invoke each compiled layer token executor;
- finalize logits if requested;
- advance session position/state.

It should not own a growing tree of mixer/FFN semantic decisions.

## 2.4 Remove semantic condition reconstruction from `forward_chunk()`

`forward_chunk()` should primarily:

- prepare chunk embeddings;
- invoke each compiled layer chunk executor;
- finalize last-row logits if requested;
- advance session positions/state.

It should not independently reproduce the layer semantic pipeline.

## 2.5 Preserve sequential-only fallback only as explicit capability

If an operator has no chunk implementation, represent that in the compiled CPU layer/operator capability rather than scanning for specific mixer identities such as Mamba2 or MLP-only.

Example:

```cpp
struct CpuLayerExecutionSpec {
    bool supports_native_chunk = true;
};
```

Better still, provide a generic sequential chunk adapter selected at compile time.

## 2.6 Token/chunk equivalence tests

For each supported semantic category compare:

- repeated token execution;
- one chunk;
- multiple chunk partitions;
- mixed chunk boundaries.

Cover:

- ordinary dense attention;
- sliding attention;
- query gate;
- Q/K norm;
- M-RoPE;
- shared KV;
- short convolution;
- GatedDeltaNet;
- Mamba2;
- dense FFN;
- MoE;
- shared expert;
- grouped top-K;
- per-layer input;
- intermediate normalization boundaries.

## 2.7 Acceptance gate

- `model_forward.cpp` is an orchestration unit rather than a second semantic implementation;
- token/chunk paths share compiled semantic ownership;
- no architecture identity is inferred from mixer presence;
- unsupported chunk behavior is an explicit compiled capability;
- chunk partitioning cannot change model semantics.

---

# Phase 3 - Remove operator dependency on the entire `CpuCompiledModel`

Priority: very high  
Risk: moderate  
Purpose: improve SRP, DIP, testability, and operator reuse.

Current CPU operator helpers still commonly receive:

```cpp
CpuCompiledModel& model
```

That gives focused operations access to far more state than they require.

## 3.1 Introduce focused execution views

Possible direction:

```cpp
struct CpuExecutionContext {
    CpuLinearEngine& linear;
    CpuThreadPool& pool;
    CpuSessionView session;
    CpuWorkspaceView workspace;
};
```

Operator-specific views may be even narrower:

```cpp
struct CpuAttentionTokenContext { ... };
struct CpuMoeChunkContext { ... };
```

Do not over-fragment purely for aesthetics. The goal is to expose only the resources required by an operator class.

## 3.2 Separate immutable shared state from mutable session state

Operators should clearly receive either:

- immutable compiled weights/program information;
- mutable per-session recurrent/KV state;
- reusable scratch;
- backend services such as thread pool/linear engine.

Avoid ambient access through a large owner object.

## 3.3 Make operator unit testing possible without full model construction

A CPU attention/MoE/recurrent test should not need to instantiate the entire runtime, tokenizer, checkpoint binding system, and model object when testing local execution semantics.

## 3.4 Acceptance gate

- operator APIs do not require unrestricted `CpuCompiledModel&` unless there is a documented exceptional reason;
- immutable weights, session state, and scratch ownership are explicit;
- focused operators can be unit-tested with small fixtures.

---

# Phase 4 - Turn `CpuWorkspace` into storage backed by a compiled workspace plan

Priority: high  
Risk: moderate  
Purpose: prevent the central workspace from knowing every operator type.

Today `CpuWorkspace::ensure()` / `ensure_chunk()` understand scratch requirements for attention, MoE, Mamba2, GatedDeltaNet, convolution, FFN, shared experts, per-layer input, and other operations.

That means adding a new operator grows the central workspace implementation.

## 4.1 Introduce `CpuWorkspacePlan`

Compile scratch requirements once from the CPU executable program.

Possible shape:

```cpp
struct CpuWorkspacePlan {
    size_t hidden_elements;
    size_t residual_elements;
    size_t norm_elements;
    size_t operator_scratch_elements;
    size_t ffn_scratch_elements;
    size_t moe_scratch_elements;
    size_t recurrent_scratch_elements;
    size_t per_layer_input_elements;
};
```

The exact representation can use named arenas, typed offsets, or grouped vectors.

## 4.2 Keep contiguous reusable storage

Do not replace the workspace with per-layer allocations.

Arenas or reusable vectors are desirable for performance. The architectural change is that their sizes and typed views are compiled from requirements rather than hard-coded around every supported operator type.

## 4.3 Typed workspace views

Operators should receive views such as:

```cpp
CpuAttentionScratch
CpuMoeScratch
CpuRecurrentScratch
```

These can reference slices of common contiguous storage.

## 4.4 Acceptance gate

- adding an operator does not require teaching a monolithic `CpuWorkspace::ensure()` its detailed shape formulas;
- steady-state inference remains allocation-free where currently expected;
- workspace memory remains observable through diagnostics.

---

# Phase 5 - Make each model family a coherent runtime extension unit

Priority: high  
Risk: low/moderate  
Purpose: complete OCP for model-family integration.

Current built-in family composition improved significantly, but family-owned contributions are still partly distributed across central tables/modules.

## 5.1 One family module factory

Each built-in family should expose a single composition entry point, for example:

```cpp
std::unique_ptr<IRuntimeModule> make_nanbeige_runtime_module();
```

or an equivalent immutable registration descriptor.

That module owns registration of all family capabilities that actually belong together.

Potential contributions:

- architecture resolver;
- chat profile;
- tool-call codec;
- tokenizer behavior/rules where family-owned;
- vision provider;
- family-specific checkpoint interpretation extensions.

## 5.2 Remove central family capability matrices

Do not maintain one global list for architecture registration, another for chat, another for tokenizer behavior, and another for vision.

The built-in composition root may still list module factories:

```cpp
make_lfm2_runtime_module()
make_granite_runtime_module()
make_nanbeige_runtime_module()
...
```

A composition root is allowed to know which built-ins ship with the binary.

It should not know the internal capabilities of each family.

## 5.3 Tokenizer behavior ownership

The generic BPE engine already moved in the right direction by consuming `TokenizerDefinition` behavior instead of model-family names.

Continue that direction.

If tokenizer behavior is truly identified by tokenizer metadata, resolve it from tokenizer metadata.

If a family must supply a compatibility rule for imperfect source metadata, that rule belongs in the family module/provider contribution rather than a global built-in tokenizer family table.

## 5.4 Vision registration ownership

Gemma/Qwen-specific vision registration should be contributed by their family modules unless there is a strong reason for an independently installable vision module.

The generic vision catalog remains generic.

## 5.5 Acceptance gate

Adding a new built-in family should normally require:

```text
src/models/<family>/**
include/celeg/models/<family>/** if public exposure is justified
one composition-root module-factory entry
build manifest entries/tests
```

It should not require editing several unrelated central capability tables.

---

# Phase 6 - Make CPU weight representations extensible before quantization formats explode

Priority: high  
Risk: moderate/high  
Purpose: prevent `variant` and dispatch growth across every linear operation.

The recent packed INT8 work exposes the next likely OCP pressure point.

Current CPU linear representation includes concrete alternatives such as:

- internal Q4;
- native GGUF matrix;
- packed INT8.

`gemv`, `gemm`, embedding, codecs, loaders, and sometimes CUDA loading logic need explicit awareness of new representations.

This will become increasingly expensive as CELEG supports more of:

- GPTQ INT4;
- GPTQ INT8 variants;
- AWQ;
- Q4/Q5/Q6 families;
- FP8;
- MXFP4;
- NVFP4;
- future compressed-tensors layouts;
- backend-native packed formats.

## 6.1 Separate source encoding from backend linear representation

Introduce an explicit binding/codec phase:

```text
checkpoint tensor representation
            |
            v
      weight codec/binder
            |
            v
 backend-native linear segment
```

The checkpoint helper may know how to decode `pack-quantized` storage.

The CPU executable should know only the CPU-native representation and operations required for execution.

## 6.2 Resolve linear dispatch once

Possible direction:

```cpp
struct CpuLinearOps {
    GemvFn gemv;
    GemmFn gemm;
    EmbeddingFn embedding;
};

struct CpuLinearSegment {
    CpuLinearOps ops;
    CpuLinearMetadata metadata;
    CpuLinearStorage storage;
};
```

Alternative tagged dispatch tables are acceptable.

The important property is that adding format N does not require adding a new chain of `holds_alternative` logic to every consumer.

## 6.3 Keep optimized special cases possible

The abstraction must still support:

- Q8 activation preparation reused across GEMMs;
- grouped GEMM;
- fused/native kernels;
- row embedding decode;
- mixed concatenation only when explicitly supported;
- backend-specific packing caches.

Do not hide useful performance capabilities behind an overly generic `void* matmul()` abstraction.

## 6.4 Move format-specific detection to codecs/binders

Calls such as `has_packed_int8_matrix()` are appropriate near binding/loading but should not become a pattern copied across execution layers.

## 6.5 Acceptance gate

- a new CPU weight representation primarily adds a codec/binding implementation and kernel implementation;
- generic `CpuLinearEngine` does not grow a new large branch tree for every representation;
- hot-path dispatch is pre-resolved;
- checkpoint format does not leak into model semantics.

---

# Phase 7 - Delete the duplicate C API and make the extensible API the only API

Priority: critical  
Risk: low/moderate because backward compatibility is explicitly not required  
Purpose: eliminate self-inflicted API duplication and restore OCP at the public boundary.

The repository currently exposes both an older CPU/CUDA-enum-based engine API and a newer backend-ID-based `v2` API.

Under CELEG's no-backward-compatibility directive, this duplication must not exist.

## 7.1 Keep the extensible design, remove the version suffix

Target public engine configuration:

```c
typedef struct celeg_engine_options {
    uint32_t struct_size;
    const char* backend_id;
    int32_t max_context;
    const void* backend_options;
    uint32_t backend_options_size;
    celeg_generation_options generation;
} celeg_engine_options;
```

Target entry point:

```c
CELEG_API celeg_engine* celeg_engine_create(
    const char* path,
    const celeg_engine_options* options);
```

Delete `celeg_engine_create_v2` after migrating repository callers.

## 7.2 Rename backend option structs without versioning

Replace names such as:

```text
celeg_cpu_backend_v2_options
celeg_cuda_backend_v2_options
```

with:

```text
celeg_cpu_backend_options
celeg_cuda_backend_options
```

## 7.3 Remove the obsolete enum/union engine-selection design

The generic public engine API should not have to add a new enum member and union arm for every backend.

Remove the old engine-selection dependency on:

```text
celeg_backend
CELEG_BACKEND_CPU
CELEG_BACKEND_CUDA
union { cpu; cuda; }
```

from the generic engine creation surface.

A backend-specific options payload remains backend-owned.

## 7.4 Delete duplicate service-bundle creation paths

Keep the path that:

```text
backend_id
  -> BackendFactoryCatalog
  -> IBackendFactory
  -> optional IBackendOptionsDecoder
  -> ServiceBundle
```

Rename it to the ordinary non-versioned API.

Delete the old `create_service_bundle()` implementation that branches directly on CPU/CUDA.

Do not preserve it as `legacy`, `v1`, or a compatibility wrapper.

## 7.5 Remove direct CPU/CUDA selection from generic engine code

After migration, generic API code should not contain logic equivalent to:

```cpp
backend == CPU ? "cpu" : "cuda"
```

Backend factories own backend-specific construction.

## 7.6 Decide separately whether `celeg_model_*` remains CPU-specific

The current lower-level model API is CPU-specific in shape. Do not accidentally preserve or version it merely because it exists.

Review it explicitly:

- if it is intentionally a low-level CPU API, name/document that clearly;
- if it is meant to be backend-neutral, refactor it to the same factory model;
- if it is obsolete relative to the engine API, delete it.

No compatibility requirement should bias the decision.

## 7.7 Acceptance gate

Repository grep should find no CELEG-owned public creation APIs with `_v2` or `_v1` suffixes.

There is one canonical engine creation path.

Adding a new backend does not require editing a public backend enum or union.

All tests/examples are migrated directly.

No compatibility wrappers remain.

---

# Phase 8 - Reduce `RuntimeContext`/builder knowledge where extension ownership can be delegated

Priority: medium  
Risk: low  
Purpose: keep dependency inversion clean as provider categories grow.

`RuntimeContext` currently owns several explicit catalogs. This is acceptable while the set is stable and semantically meaningful, but avoid turning it into a universal service locator.

## 8.1 Keep typed catalogs for major stable extension categories

Typed catalogs are desirable for:

- architectures;
- checkpoint formats;
- backend factories;
- tokenizer providers;
- chat profiles;
- vision providers.

Do not replace clear typed contracts with `std::any` or stringly-typed service lookup.

## 8.2 Resist adding every future capability directly to `RuntimeContext`

Before adding a new catalog, decide whether it belongs under an existing family/runtime module or backend-local registry.

Examples:

- a CUDA kernel policy does not belong in global runtime context;
- a family-only decoder helper does not belong in global runtime context;
- an internal CPU linear codec does not belong in global runtime context.

## 8.3 Acceptance gate

`RuntimeContext` remains a composition boundary for true cross-cutting extension categories rather than a dependency bag.

---

# Phase 9 - Consolidate validation around ownership boundaries

Priority: medium  
Risk: low  
Purpose: make invalid states fail before serving without duplicating validation everywhere.

## 9.1 Architecture validation

Architecture resolution validates:

- metadata assumptions;
- graph semantic completeness;
- weight-plan completeness;
- family-specific unsupported variants.

## 9.2 Generic semantic validation

`ModelGraph` validates:

- per-layer semantic consistency;
- dimensions;
- attention properties;
- recurrent properties;
- FFN/MoE properties;
- normalization requirements.

## 9.3 Compiled-program validation

`CompiledModelProgram` validates:

- every logical layer compiled;
- compiled indices valid;
- required semantic specs present;
- MoE program internally valid;
- no checkpoint/model-family identity leaked into execution.

## 9.4 Backend capability validation

CPU/CUDA compilers validate whether they support the compiled semantics before allocation/serving.

Examples:

- grouped MoE selection;
- shared experts;
- payload layout;
- M-RoPE;
- native chunk support;
- required recurrent mixer;
- quantized storage representation.

## 9.5 Avoid validation duplication

Do not validate the same semantic rule in architecture, graph, compiled program, CPU load, and CUDA load unless the checks protect genuinely different boundaries.

---

# Phase 10 - Reassess public/internal header ownership

Priority: medium  
Risk: low/moderate  
Purpose: keep compile-time dependencies and architectural boundaries clear.

## 10.1 Generic public headers

Public generic headers should contain contracts and semantic types needed by consumers.

Do not expose backend implementation internals merely because multiple translation units need them.

## 10.2 Backend detail headers

Large structures such as CPU weight stores, scratch internals, caches, and session implementation state should remain private/detail where possible.

## 10.3 Family headers

Only expose family headers publicly when external code genuinely needs to construct or inspect that family-specific feature.

Registration details should generally remain implementation-level.

## 10.4 Acceptance gate

The public include tree communicates stable architectural contracts, not internal convenience dependencies.

---

# Phase 11 - Documentation and architecture evidence cleanup

Priority: medium  
Risk: low

## 11.1 Update architecture evidence after each completed phase

`docs/ARCHITECTURE_EVIDENCE.md` should document actual enforced boundaries, not aspirational ones.

## 11.2 Mark the previous plan as superseded

Update `CELEG_SOLID_EXECUTION_AND_TOKENIZER_REFACTORING_PLAN.md` with a short pointer to this file once implementation work starts, or delete it if retaining historical plans has no value.

Given the no-compatibility directive, do not keep obsolete documentation merely to preserve old paths.

## 11.3 Keep evidence executable

Prefer evidence in:

- tests;
- static dependency checks;
- compile-time boundaries;
- architecture-resolution fixtures;
- token/chunk equivalence tests.

Documentation alone must not be the only enforcement mechanism.

---

# 12. Recommended execution order

The recommended order is deliberately chosen to reduce correctness risk.

## Stage A - Guardrails

1. Replace family-name-based architecture checker with structural dependency checks.
2. Add no-compatibility-debt checks.
3. Strengthen fake family/backend extension tests.
4. Add/confirm token-vs-chunk equivalence tests.

## Stage B - Semantic source of truth

5. Make `ModelGraph` canonical.
6. Derive runtime shape from graph.
7. Move checkpoint physical-layer mapping into binding/weight planning.
8. Compile `CompiledModelProgram` from canonical semantic state only.

## Stage C - CPU execution architecture

9. Introduce CPU compiled layer executors.
10. Collapse token/chunk semantic orchestration duplication.
11. Replace full-model operator dependencies with focused execution contexts.
12. Introduce compiled workspace planning.

## Stage D - Extension completeness

13. Make each family a coherent runtime module.
14. Move tokenizer/vision family contributions into family ownership where appropriate.
15. Make CPU weight representation dispatch extensible.

## Stage E - Public API cleanup

16. Delete old enum/union engine API.
17. Rename the backend-ID API to the only `celeg_engine_*` API.
18. Remove all `_v2` CELEG-owned public engine symbols.
19. Migrate all examples/tests/callers directly.
20. Review whether the lower-level `celeg_model_*` surface should remain, be generalized, or be deleted.

## Stage F - Final cleanup

21. Update architecture evidence.
22. Remove obsolete helpers/types/includes.
23. Run full CPU test suite.
24. Build CUDA targets.
25. Run CUDA tests where a device is available.
26. Run static boundary checks.
27. Re-score SOLID and DRY after completion.

---

# 13. Explicit anti-patterns to reject during this refactor

Do not solve the remaining architecture debt with any of the following.

### 13.1 Architecture switches in backends

Bad:

```cpp
if (architecture_id == "nanbeige42") { ... }
```

Correct direction: compile explicit semantic properties before backend execution.

### 13.2 Format switches in model semantics

Bad:

```cpp
if (checkpoint_is_gptq) graph.layers[i]....
```

Storage format belongs to binding/codecs, not model semantics.

### 13.3 One interface per trivial function

SOLID does not require wrapping every free function in a virtual class.

Prefer value semantics and compiled records when stateful polymorphism is unnecessary.

### 13.4 Service locator growth

Do not pass `RuntimeContext` into every low-level operation as a way to avoid explicit dependencies.

### 13.5 Workspace fragmentation

Do not replace one oversized workspace with hundreds of tiny per-operation allocations.

Use compiled storage plans and typed views.

### 13.6 Public API version stacking

Bad:

```text
celeg_engine_create
celeg_engine_create_v2
celeg_engine_create_v3
```

Current project policy is replacement, not compatibility layering.

### 13.7 Transitional aliases left indefinitely

Bad:

```cpp
using OldType = NewType;
```

when the only purpose is preserving an obsolete name.

Migrate callers and delete the old name.

### 13.8 Family-name static checker lists

A checker that must learn every future model name is not enforcing an architectural boundary; it is maintaining a blacklist.

---

# 14. Definition of done

This plan is complete when the following are true.

## Semantic architecture

- `ModelGraph` is the canonical model semantic representation.
- Runtime allocation shape is derived from canonical semantics.
- `CompiledModelProgram` does not merge competing semantic sources.
- checkpoint physical storage mapping is separated from executable semantics.

## CPU execution

- token and chunk execution consume the same compiled layer semantics.
- `model_forward.cpp` is primarily orchestration.
- operators do not depend on unrestricted `CpuCompiledModel&` by default.
- workspace requirements are compiled rather than centralized by operator type.
- token/chunk equivalence and chunk-boundary invariance are covered.

## Extensibility

- new architecture families are coherent modules.
- generic runtime does not know family capability details.
- backend registration is open without core modification.
- adding a new weight representation does not require editing every linear operation.

## API

- one engine API exists.
- no `celeg_engine_create_v2` remains.
- no CELEG-owned public `_v2` backend option types remain.
- engine backend selection is string/factory based.
- generic engine API has no CPU/CUDA union that must grow for every backend.
- no compatibility shim exists solely to preserve superseded CELEG APIs.

## Enforcement

- static checker uses structural dependency boundaries, not family-name blacklists.
- no-compatibility-debt checks exist.
- architecture/backend extension tests exist.
- architecture evidence matches implemented code.

## Performance

- no architecture or checkpoint-format probing occurs in token decode.
- no refactor introduces per-token/per-layer heap allocation in steady state.
- dispatch that can be resolved at load time is resolved at load time.
- CPU/CUDA benchmark regressions are measured before accepting structural changes to hot paths.

---

# 15. Target architecture summary

```text
                              Runtime composition
                                     |
          +--------------------------+--------------------------+
          |                          |                          |
          v                          v                          v
   Family modules             Backend factories          Checkpoint formats
          |                          |                          |
          v                          |                          v
 Architecture resolution             |                 repositories/views
          |                          |                          |
          +-------------+------------+--------------------------+
                        |
                        v
                 canonical ModelGraph
                 policies + WeightPlan
                        |
              +---------+---------+
              |                   |
              v                   v
        derived shape      CompiledModelProgram
              |                   |
              +---------+---------+
                        |
          +-------------+-------------+
          |                           |
          v                           v
     CPU compiler                 CUDA compiler
          |                           |
          v                           v
  CpuCompiledLayer[]          Cuda executable plan
          |                           |
    +-----+------+                    |
    |            |                    |
 token exec   chunk exec              |
    |            |                    |
    +-----+------+                    |
          |                           |
          v                           v
     backend kernels             backend kernels
```

Checkpoint encoding enters through repository/binding/codecs and terminates in backend-native weights. Architecture identity terminates at semantic resolution/provenance. The execution loop sees neither model-family identity nor source-format identity.

---

# 16. Final architectural principle

The central rule for future CELEG development should be:

> **Interpret identity once, compile semantics once, bind storage once, and execute only the compiled result.**

A new model family should primarily add semantic resolution. A new checkpoint encoding should primarily add binding/codec support. A new backend should primarily add a backend compiler/executor. None of those extensions should force unrelated central switches to grow.

And until CELEG declares an explicit compatibility milestone:

> **Replace obsolete designs; do not version them.**
