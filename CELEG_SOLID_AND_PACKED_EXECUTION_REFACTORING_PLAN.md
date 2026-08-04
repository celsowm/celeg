# Celeg SOLID, DRY, Packed Execution, Expert Residency, and Multi-MoE Refactoring Plan

**Repository:** `celsowm/celeg`  
**Target branch:** current default branch (`master`)  
**Reviewed code head:** `79fdba3347cdec14b23a8efd3eb8166e486fbe6f`  
**Previous plan commit:** `d1af6567d42031c9ac85736d93d824fccee3b6cd`  
**Review date:** 2026-08-04  
**Primary goal:** finish the SOLID and DRY refactoring while preserving Celeg's CPU/CUDA performance model, making the SSD → RAM → VRAM expert hierarchy correct and bounded, and introducing a model-neutral compiled MoE contract so additional MoE families do not require architecture checks or LFM-specific assumptions inside the backends.

---

## Role

Act as a senior C++/CUDA inference-runtime architect working directly on the Celeg repository.

This is an implementation plan, not a request for another architecture report. Verify each finding against the current branch, implement the work in coherent stages, add regression tests before changing confirmed behavior, and leave the repository buildable after every completed stage.

Do not reintroduce architecture enums, model variants, checkpoint-format branching in backends, concrete service construction in generic APIs, or virtual dispatch inside transformer inner loops.

---

## Architectural destination

The intended dependency and compilation flow is:

```text
checkpoint formats
        ↓
family-owned architecture resolution
        ↓
neutral ResolvedModel + semantic layer programs
        ↓
CPU/CUDA backend compilation and binding
        ↓
backend-specific executable plans
        ↓
preallocated execution, residency, and serving
```

For MoE layers specifically:

```text
family metadata and tensor naming
        ↓
neutral compiled MoE semantics
        ↓
backend lowering
        ↓
router execution → selected ExpertKey set
        ↓
expert source/cache/residency
        ↓
expert kernels and output combination
```

The family owns **what the model means**. The backend owns **how those semantics execute**. The residency subsystem owns **where the required expert payload currently lives**. None of these layers should identify a model family by name.

Preserve the core execution strategy:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

For storage-backed experts, extend it with:

```text
index once
read on demand
lease while active
publish only after transfer
release after completion
```

---

## What changed since the earlier plan

The repository now contains a substantial expert-storage subsystem:

- CUDA SSD backing with a shared host cache and per-layer VRAM residency;
- active-FFN pinning and probation-slot reservation;
- frequency-aware host and device cache policies;
- an asynchronous expert I/O worker pool;
- CPU disk-backed experts loaded lazily from `.lfmpack`;
- concurrent indexed CPU pack reads;
- shared CPU expert caches across cloned sessions;
- streaming pack creation that does not retain every expert payload in RAM;
- tests proving lazy pack payload access and basic cache behavior;
- documentation for CPU and CUDA SSD tiers.

These capabilities are valuable, but the current design still assumes parts of the LFM expert representation in supposedly reusable infrastructure. Decoupling cache and residency alone is insufficient: a second MoE family also needs a neutral description of routing, expert tensor layout, optional shared experts, execution, and output combination.

The plan therefore adds a first-class **model-neutral compiled MoE program** before completing the source/cache/residency refactoring.

---

## Definition of success

The work is complete only when all of the following are true:

1. a new family registers architecture, chat, tools, tokenizer, vision, and other family-owned providers through one coherent module boundary;
2. generic runtime composition does not include, construct, or name concrete model families;
3. each MoE layer is represented by an immutable neutral semantic program produced during architecture resolution or compilation;
4. CPU and CUDA backends consume that neutral MoE program without checking an architecture identity;
5. LFM-specific `gate_up`/`down`, stacked-tensor, routing, normalization, bias, or scaling assumptions do not leak into neutral cache/residency contracts;
6. standard routed top-K, optional shared experts, grouped routing, expert bias, routing normalization, routed scaling, and different expert payload layouts can be represented without adding family flags;
7. an unsupported MoE semantic operation is rejected during backend compilation with a useful diagnostic, not discovered inside decode;
8. backend selection returns a fully constructed service bundle through a real factory;
9. the C API translates ABI data but does not construct concrete CPU or CUDA services;
10. packed decode and ragged prefill have distinct workflow owners rather than pass-through wrappers around a god executor;
11. SSD source access, host caching, GPU residency, transfer scheduling, and usage metrics have explicit owners;
12. a generic runtime cache does not depend on SafeTensors, CUDA, or a particular expert payload layout;
13. active expert payloads cannot be evicted until CPU GEMVs or CUDA transfers and FFN execution have completed;
14. the all-resident CUDA expert path performs no disk I/O, heap allocation, worker-pool submission, full-selection D2H copy, or unconditional stream synchronization;
15. cold expert admission is bounded by configured RAM/VRAM budgets and fails before publishing partial residency state;
16. CPU `.lfmpack` opening remains metadata-only and payloads remain lazy until a routed expert is acquired;
17. cache hit, true miss, coalesced wait, eviction, storage read, H2D transfer, and wait-time metrics have documented semantics;
18. shared semantic rules have one authoritative implementation where they genuinely share reasons to change;
19. CPU and CUDA mechanisms remain separate where their storage, synchronization, quantization, and kernel behavior differ;
20. no architecture probing, repeated tensor lookup, plan construction, or dynamic polymorphic graph traversal is introduced into hot paths;
21. tests and architecture checks cover each extracted boundary and failure transition;
22. CI reporting distinguishes compile-only CUDA validation from runtime validation on a real GPU;
23. final documentation describes implemented contracts and extension paths rather than intentions.

Target assessment after completion:

| Area | Target |
|---|---:|
| Single Responsibility | 8.4+ |
| Open/Closed | 8.7+ |
| Liskov Substitution | 8.5+ |
| Interface Segregation | 8.8+ |
| Dependency Inversion | 8.8+ |
| DRY | 8.4+ |
| Multi-MoE extensibility | 8.6+ |
| Packed execution | 8.2+ |
| Expert residency/storage | 8.3+ |
| Overall architecture | 8.6+ |

---

## Preserve these improvements

Do not undo:

- `IArchitecture`, `ArchitectureCatalog`, architecture probing, and backend-neutral `ResolvedModel` resolution;
- neutral checkpoint tensor and repository contracts;
- segregated repository capabilities for location, random access, native blocks, and tokenizer data;
- family-owned tensor naming policies;
- explicit token, numerical, execution, and offload policies;
- immutable compiled model programs and execution plans;
- CPU and CUDA compiler boundaries;
- prebound or precompiled hot-path decisions;
- `IRequestService`, `ISchedulerDriver`, and `IServiceDiagnostics` as separate serving roles;
- host commit occurring only after successful CUDA completion;
- packed steady-state preallocation and zero-allocation expectations;
- architecture boundary automation;
- native GGUF memory mapping and OS page-cache behavior;
- direct streaming writes during CPU pack construction;
- compact indexed CPU pack readers that do not load payloads while opening;
- concurrent indexed pack reads;
- CPU cache sharing across cloned sessions;
- coalescing concurrent loads of the same expert;
- CUDA active-batch pinning;
- CUDA probation-slot reservation for the complete unique cold set;
- device-side cold-set discovery;
- failure before host-visible state commit;
- explicitly closed and exhaustively tested domains.

---

## DRY policy

DRY means **one authoritative representation of knowledge**, not zero similar lines.

Actionable shared knowledge currently includes:

- expert key identity;
- payload-region naming and offsets after a layout is compiled;
- cache metric definitions;
- heat decay and frequency/recency scoring only where semantics are identical;
- packed compatibility identity;
- common packed-session validation;
- JSON escaping;
- tagged tool-call block lifecycle;
- standard dense-decoder topology construction;
- family registration across runtime catalogs.

Do not force one generic implementation over mechanisms with independent reasons to change:

- CPU quantized expert objects;
- raw host staging payloads for CUDA transfers;
- per-layer VRAM slot and pointer-table management;
- native GGUF mappings;
- BF16, INT8, INT4, GGUF, DP4A, tensor-core, or other materially different kernels;
- CPU mutex/future coordination and CUDA stream/event coordination;
- family probe rules and exceptional metadata semantics;
- tensor naming policies for different checkpoint conventions;
- intentionally closed public ABI translation.

Before extracting shared code, answer:

1. Is this the same knowledge or merely similar syntax?
2. Must all copies change together when the rule changes?
3. Can the contract be named without mentioning unrelated callers or families?
4. Does the extraction reduce parameters and conditionals rather than move them?
5. Can it be tested without constructing a complete model/backend?
6. Does it preserve hot-path allocation and dispatch behavior?

If any answer is negative, keep the mechanisms separate and document the intentional difference.

---

# Current high-priority findings

## 1. MoE semantics are not yet represented by a model-neutral compiled program

### Problem

The current executable representation contains useful neutral topology, but the MoE path still exposes assumptions tied to the presently supported LFM layout and execution shape. Expert payload interpretation, router configuration, routed expert execution, residency pointer tables, and backend descriptors are too close to one concrete arrangement.

A new MoE family may differ in one or more of the following:

- router score calculation;
- top-K selection and normalization;
- grouped or constrained expert selection;
- expert bias and correction bias;
- routed output scaling;
- shared experts executed beside routed experts;
- number of experts or experts-per-token by layer;
- expert activation and MLP structure;
- fused, stacked, individual, or differently ordered tensors;
- multiple payload regions per expert;
- quantization/storage encoding;
- combination order between routed and shared output.

Without a neutral semantic program, adding such a family risks backend architecture switches or an expanding structure full of LFM defaults and family-specific booleans.

### Required direction

Represent every MoE layer as immutable neutral semantics before backend binding. The exact names may differ, but the shape should resemble:

```cpp
struct ExpertPayloadRegion {
    TensorRole role;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    TensorDType dtype = TensorDType::Unknown;
};

struct ExpertPayloadSchema {
    std::size_t total_bytes = 0;
    std::vector<ExpertPayloadRegion> regions;
};

struct RouterProgram {
    RouterScoreProgram score;
    ExpertSelectionProgram selection;
    RoutingNormalizationProgram normalization;
    std::optional<ExpertBiasProgram> bias;
};

struct RoutedExpertProgram {
    ExpertMlpProgram mlp;
    ExpertPayloadSchema payload;
};

struct SharedExpertProgram {
    ExpertMlpProgram mlp;
    SharedExpertCombineProgram combine;
};

struct MoeLayerProgram {
    RouterProgram router;
    RoutedExpertProgram routed;
    std::optional<SharedExpertProgram> shared;
    MoeOutputProgram output;
    ExpertResidencyRequirements residency;
};
```

These are conceptual contracts, not mandatory exact types. Prefer compact value types, tagged semantic operations, and immutable spans owned by the compiled model. Do not create a virtual object graph or a mega-structure with nullable fields for every known family.

The architecture/family resolver must translate checkpoint metadata into this neutral program. CPU and CUDA compilers must lower it into backend-specific executable descriptors and function choices. The hot path consumes only the lowered program.

### Required boundaries

- tensor naming and metadata interpretation remain family-owned;
- the neutral program contains semantic roles, dimensions, policies, and compiled payload regions—not checkpoint tensor names;
- residency receives `ExpertKey` plus an immutable payload schema/manifest, not architecture identity;
- router execution produces selected expert identities independently of where weights reside;
- source adapters materialize the payload required by the compiled schema;
- backend lowering selects kernels once and rejects unsupported operations before session creation;
- packed and standalone paths consume the same lowered MoE program;
- no `if (family == ...)`, architecture enum, model-name string, or checkpoint-name suffix appears in backend execution.

### Acceptance criteria

- existing LFM MoE resolves into the neutral program and remains numerically equivalent;
- a synthetic ordinary top-K routed MoE compiles without LFM names;
- a synthetic MoE with a shared expert compiles and validates output combination order;
- a synthetic grouped-routing program validates selection constraints;
- payload tests cover stacked and individually located expert tensors;
- layer-local expert counts, widths, and experts-per-token are representable or explicitly rejected during compilation;
- CPU and CUDA compilers report unsupported semantic operations before allocating execution state;
- backend tests use synthetic neutral programs rather than named architectures;
- adding a compatible MoE family requires family-owned resolution/naming plus registration, not edits to generic backend orchestration;
- program fingerprinting includes every execution-relevant MoE semantic and payload-layout field;
- no architecture identity is required by expert cache, source, residency, router, or FFN execution.

---

## 2. CUDA expert residency has become a second god orchestrator

`SharedModelWeights::ensure_moe_experts_resident` currently coordinates locking, event dependencies, cold discovery, transfer cleanup, metrics, source selection, disk reads, cache acquisition, future waits, H2D promotion, publication, resident touching, and prefetch.

### Required direction

Introduce typed collaborators:

```cpp
struct ExpertResidencyRequest;
struct ExpertResidencyWorkspace;
class IExpertSource;
class IHostExpertCache;
class ExpertLayerResidency;
class CudaExpertResidencyCoordinator;
```

- `ExpertLayerResidency` owns VRAM slots, pointer tables, protection, and device metadata;
- `IExpertSource` reads the payload described by the compiled schema;
- `IHostExpertCache` owns host admission, coalescing, leases, and eviction;
- a bounded I/O executor owns workers and queue behavior;
- the coordinator owns one discover/acquire/transfer/publish/complete transaction;
- metrics receive explicit events rather than scattered mutations.

### Acceptance criteria

- no residency method takes a parameter maze of events and scratch vectors;
- source selection is outside transfer orchestration;
- VRAM code does not read checkpoint formats;
- host cache does not publish CUDA pointer tables;
- failed read/transfer leaves the previous residency map valid;
- active FFN slots remain protected through completion;
- fake source/cache/residency collaborators can test the coordinator;
- packed and standalone paths share one residency algorithm.

---

## 3. `PinnedExpertCache` is in the wrong abstraction layer

The supposedly generic runtime cache includes a concrete checkpoint-format header, hardcodes `gate_up` and `down`, combines storage, lease behavior, frequency tracking, coalescing, metrics, and I/O execution, and is called “Pinned” although pinning is allocator-dependent.

### Required direction

Either:

1. make it an explicitly CUDA-owned host staging cache; or
2. make it a neutral host blob cache driven by `ExpertKey`, `ExpertPayloadSchema`, and an explicit allocator.

Keep family payload interpretation in family/backend binding, format access in source adapters, and worker lifetime in an I/O component.

### Acceptance criteria

- generic runtime headers include no concrete checkpoint format;
- neutral cache code names no LFM tensor region;
- allocator/pinning behavior is explicit;
- I/O lifetime is independent of cache storage lifetime;
- focused cache builds require neither CUDA nor format headers.

---

## 4. Cache policy, capacity, and metric semantics are duplicated and drifting

CPU RAM cache, host staging cache, and GPU layer cache encode related but not identical policy knowledge. Key packing, decay constants, scoring, promotion, and hit/miss semantics can diverge accidentally.

### Required direction

Do not create `ExpertCache<T, 20 callbacks>`. Extract only stable value/policy contracts:

- `ExpertKey` and hashing;
- counter definitions;
- budget/admission results;
- policy configuration validation;
- decay/scoring primitives only where semantics are truly the same.

Keep storage, locking, transfer, and eviction mechanisms tier-specific.

### Acceptance criteria

- `true_misses` means the elected loader;
- `coalesced_waits` is separate;
- undersized and zero budgets have explicit tested behavior;
- protected/probation differences are documented by tier;
- deterministic traces test scoring;
- comparable metrics do not pretend mechanisms are identical.

---

## 5. CUDA residency violates or obscures the hot-path contract

The path currently contains unconditional synchronization, full-selection D2H copies, per-miss vector/future construction, synchronous waiting, and host loops over selected experts.

### Required direction

**All-resident path:** no allocation, worker submission, disk read, full-selection D2H copy, or unconditional synchronization.

**Cold path:** one compact unique cold list, bounded preallocated metadata, capped parallel I/O, no global lock during reads, event-based H2D completion, and transactional publication.

### Acceptance criteria

- repeated all-resident decode performs zero host allocations;
- no `cudaStreamSynchronize` exists in the all-resident path;
- only policy-required data crosses D2H;
- miss work is preallocated or explicitly bounded;
- prefetch cannot delay or invalidate current-token admission;
- hit/miss timing is measured separately.

---

## 6. Expert lease, transfer, publication, and shutdown invariants are implicit

Correctness spans host leases, inflight transfers, events, batch-pinned slots, loading generations, worker shutdown, source lifetime, and shared model lifetime.

### Required direction

Model explicit transitions:

```text
empty -> loading -> host-resident -> transferring -> device-resident
                                      -> failed

device-resident -> active -> device-resident
                -> evicted
```

Use RAII tickets holding the required source/cache/slot resources until completion is observed. Publish state transactionally.

### Acceptance criteria

- duplicate release and reference underflow are detected;
- worker/queue/budget/payload configuration is validated;
- shutdown drains or cancels according to a documented rule;
- all coalesced waiters receive failures;
- retries start from clean state;
- failed H2D never publishes a device pointer;
- host payload and device slot live through their completion events;
- destruction order is tested.

---

## 7. CPU disk-backed expert ownership is spread across compiled-model internals

Configuration, pack reader, cache, weight-store mutation, naming, validation, and acquisition are distributed through `CpuCompiledModel::Shared` and forward code.

### Required direction

Introduce a CPU-owned backing boundary:

```cpp
class CpuExpertBackingStore {
public:
    std::shared_ptr<const CpuExpertWeights> acquire(ExpertKey key);
    CpuExpertBackingMetrics metrics() const;
};
```

It may own a pack reader, cache, payload/name resolver, and immutable shape validation. Native GGUF remains a separate mapped path.

### Acceptance criteria

- forward code requests an expert lease without pack details;
- pack naming and dimensions have one owner;
- opening an existing pack reads no payload;
- first construction writes and releases one expert at a time;
- cloned sessions share the backing store intentionally;
- native GGUF avoids a redundant cache;
- different experts can be read concurrently.

---

## 8. Packed execution still concentrates too many responsibilities

`PackedDecodeExecutorImpl` still owns shared state, compatibility, validation, staging, GEMM setup, embedding, layer traversal, attention, convolution, dense FFN, MoE, completion, commit, and metrics. Decode/prefill pipeline wrappers remain mostly pass-through.

### Required direction

- decode orchestration belongs to `PackedDecodePipeline`;
- ragged prefill orchestration belongs to `PackedPrefillPipeline`;
- shared preallocated resources live in an explicit workspace/resource object;
- both consume the same lowered MoE program and residency coordinator as standalone execution;
- validation, staging, completion/commit, and metrics have narrow owners.

### Acceptance criteria

- pipeline `run()` methods own operation sequences;
- shared state does not know both complete algorithms;
- MoE lowering/residency is not duplicated;
- failed CUDA work cannot mutate host-visible state;
- parity and zero-allocation tests remain green.

---

## 9. `IBackendFactory` is not a real factory

The abstraction identifies/supports backends while generic composition still constructs concrete services.

### Required direction

A selected factory must own concrete creation from a neutral request and return serving roles as a bundle.

### Acceptance criteria

- `celeg_engine_create` constructs no concrete inference service;
- adding a backend adds no C API branch;
- unsupported builds fail before partial mutation;
- ABI mapping remains outside backend implementation;
- CPU-only linkage references no CUDA implementation.

---

## 10. Generic runtime composition still knows concrete families

Generic runtime bootstrap still contains family knowledge, including concrete vision construction.

### Required direction

Introduce a family module boundary:

```cpp
class IRuntimeModule {
public:
    virtual ~IRuntimeModule() = default;
    virtual std::string_view id() const = 0;
    virtual void register_into(RuntimeBuilder& builder) const = 0;
};
```

A module may register architecture resolution, chat profile, tool codec, tokenizer, vision, and other cold-path providers.

### Acceptance criteria

- each family has one registration entry point;
- generic runtime contains no family include/string/type/constructor;
- duplicate registration is rejected;
- catalogs freeze after construction;
- custom runtimes can include or omit built-ins.

---

## 11. Family registration knowledge remains duplicated

Supported-family knowledge is repeated across architecture, chat, tools, tokenizer, vision, source lists, and tests.

### Required direction

Make family contributions atomic through modules without a central nullable mega-structure.

### Acceptance criteria

- adding a family requires one built-in module entry plus family-owned files;
- families without optional capabilities implement no fake methods;
- module tests verify all expected contributions;
- partial source/build registration is guarded where practical.

---

## 12. Dense decoder topology construction is duplicated

MiniCPM5 and SmolLM3 repeat standard dense topology and paired GGUF/JSON access.

### Required direction

Extract a neutral dense-decoder topology builder driven by metadata mappings and defaults. Probe identity and real exceptions remain family-owned.

### Acceptance criteria

- standard invariants have one implementation;
- no family names/flags enter the builder;
- GGUF and JSON paths have focused tests;
- fingerprints remain stable unless fixing a demonstrated bug.

---

## 13. JSON and tagged tool-call parsing have multiple implementations

Extract small protocol-neutral utilities for JSON escaping and tagged-block lifecycle. Use the existing real JSON parser for arbitrary JSON.

### Acceptance criteria

- one escaping implementation covers controls, UTF-8, quotes, slashes, and empty strings;
- shared tagged-block scanning/removal has one implementation;
- protocol syntax and malformed-output rules remain family-specific;
- mixed text and parallel calls are preserved.

---

## 14. Chat templates and tool codecs overlap in responsibility

Move family templates/codecs under family modules. Give native tool syntax one owner, preferably the codec, while the conversation template owns role sequencing and placement.

### Acceptance criteria

- generic chat code has no family formatting rules;
- tool syntax is not independently rebuilt twice;
- prompt fixtures remain byte-identical unless documenting a bug fix.

---

## 15. Packed common validation is duplicated

Extract common batch invariants and layer decode/prefill-specific validation on top. Invalid input must fail before CUDA work or host mutation.

---

## 16. Packed compatibility knowledge is duplicated

Produce an immutable compatibility key during compilation. It must include execution-relevant MoE semantics/fingerprint, expert source/residency compatibility, device/resource identity, and ordinary attention/KV/GEMM options without including observability-only settings.

---

## 17. C API CPU/CUDA composition remains duplicated

Fix real factories first. C API code should only validate struct sizes/enums and translate public values into neutral configuration.

---

# SOLID-specific requirements

## Single Responsibility Principle

A valid extraction moves ownership, lifetime, dependency direction, invariant enforcement, state-transition authority, synchronization authority, compilation boundary, or a directly testable contract. Moving methods to another `.cpp` is insufficient.

Prioritize:

- neutral MoE semantics and backend lowering;
- CUDA residency orchestration;
- host cache and I/O execution;
- CPU expert backing;
- packed executor ownership;
- generic runtime bootstrap;
- C API composition.

## Open/Closed Principle

Adding a compatible MoE family must not require edits to generic CPU/CUDA execution, cache, residency, packed execution, serving, or C API layers. Adding a new semantic operation may require a new backend lowering implementation, but not a named-family case.

Adding an expert source must not rewrite cache eviction or VRAM slot management. Adding a backend must not add a C API construction branch.

## Liskov Substitution Principle

- neutral MoE programs with supported semantics must lower consistently regardless of family origin;
- unsupported programs must fail during compilation before partial mutation;
- source implementations must honor concurrent-read and lifetime contracts;
- caches must honor advertised lease/failure semantics;
- asynchronous completion must define when payloads and pointer tables are observable;
- family modules must work in custom builders without hidden globals.

## Interface Segregation Principle

Do not create one god interface combining router, source, cache, residency, kernels, and metrics. A family without tools/vision/shared experts, a source without random access, or a cache without async loading must not implement placeholders.

## Dependency Inversion Principle

Forbidden directions include:

```text
src/runtime/**                 -> celeg/models/**
src/runtime/cache/**           -> checkpoint/formats/**
include/celeg/model/**         -> celeg/backend/cuda/**
src/backend/**                 -> architecture identity dispatch
src/backend/**                 -> concrete checkpoint-format dispatch
backend MoE execution          -> model-family identity
VRAM residency                -> router semantics or format parsing
host cache                    -> CUDA pointer publication
src/api/**                    -> concrete inference service construction
```

---

# Implementation stages

Keep every stage independently buildable and testable.

## Stage 0 — Re-baseline and protect behavior

- record current head and changes since the reviewed code head;
- run architecture checks and available builds/tests;
- capture prompt fixtures and numerical parity;
- capture existing LFM MoE resolved topology/program fingerprints;
- capture CPU pack/cache and CUDA hit/miss/all-resident performance;
- record allocations, D2H copies, synchronization, and which CUDA tests actually ran on a GPU.

## Stage 1 — Lock down current expert and MoE semantics

Add tests for:

- existing router top-K, normalization, bias, scaling, expert layout, and output combination;
- cache metric definitions and budget edges;
- lease and active-FFN eviction protection;
- failure/retry/publication transitions;
- native GGUF bypass and pack laziness;
- packed/standalone MoE parity.

Fix confirmed defects before restructuring.

## Stage 2 — Introduce the neutral compiled MoE program

- define neutral semantic value types and per-layer MoE program ownership;
- compile existing LFM metadata/tensor roles into the new program;
- fingerprint every execution-relevant field;
- remove checkpoint tensor names from executable semantics;
- add synthetic programs for ordinary top-K, shared expert, grouped routing, and alternate payload layouts;
- add explicit backend capability validation.

Do not change expert residency or kernels until equivalence tests prove the new representation.

## Stage 3 — Lower neutral MoE programs in CPU and CUDA compilers

- bind tensor roles/payload regions once;
- select router, expert, shared-expert, and combine implementations once;
- produce compact backend-specific executable descriptors;
- reject unsupported operations during compilation;
- remove named-family and suffix-based MoE decisions from backend code.

## Stage 4 — Extract neutral expert policy and metric values

Extract key/hash, counter definitions, budget/admission values, configuration validation, and only genuinely shared scoring math. Keep tier mechanisms separate.

## Stage 5 — Separate expert source, host cache, and I/O executor

- remove format includes and LFM regions from neutral cache code;
- make payload schemas/manifests explicit;
- add sidecar, pack, and random-access source adapters;
- separate bounded executor lifetime from cache lifetime;
- make pinned-memory allocation explicit.

## Stage 6 — Refactor CUDA residency orchestration

- typed request and preallocated workspace;
- separate VRAM slot state from transaction orchestration;
- centralize lease/event lifetime;
- transactional publication;
- compact cold discovery;
- no unconditional hit-path synchronization;
- prefetch subordinate to correctness;
- shared use by standalone and packed execution.

## Stage 7 — Encapsulate CPU expert backing

Centralize pack source, naming/manifest resolution, validation, cache ownership, acquisition, and metrics while preserving lazy reads, streaming construction, concurrent indexed access, cloned-session sharing, and native GGUF mapping.

## Stage 8 — Shared low-risk DRY utilities

Implement JSON escaping, tagged-block lifecycle, identical metadata helpers, common packed validation, and immutable packed compatibility.

## Stage 9 — Dense decoder topology builder

Extract only standard dense invariants from naturally compatible families.

## Stage 10 — Family-owned chat/template/tool modules

Move family text protocols under their modules and remove duplicate tool syntax ownership.

## Stage 11 — Runtime family modules

Remove family knowledge from generic runtime bootstrap and make built-in contributions atomic.

## Stage 12 — Real backend factories and neutral C API

Factories create service bundles; C API only translates/validates ABI values.

## Stage 13 — Packed execution ownership

Move decode and ragged-prefill workflows into real pipeline owners using shared lowered MoE programs and the common residency coordinator.

## Stage 14 — Expand architecture, DRY, and hot-path guards

Add stable checks for:

- family dependencies in generic runtime;
- format dependencies in generic cache;
- architecture identity dispatch in backends;
- model-family names in MoE execution/residency;
- checkpoint tensor-name decisions after backend binding;
- direct source reads from VRAM residency;
- concrete backend construction from C API;
- pass-through packed pipelines;
- allocation/plan construction in steady-state paths.

## Stage 15 — Full verification and documentation

Run the full matrix, publish benchmark/allocation evidence, and update `docs/EXPERT_STORAGE.md`, `docs/PACKED_EXECUTION.md`, and a new or existing architecture document describing how to add another MoE family through the neutral program.

---

# Testing requirements

## Neutral MoE program and backend lowering

- existing LFM resolution equivalence;
- synthetic ordinary top-K routed layer;
- normalization enabled/disabled;
- expert bias enabled/disabled;
- routed scaling;
- shared expert before/after routed combination as supported semantics;
- grouped selection constraints;
- varying expert counts, top-K, and intermediate widths by layer;
- stacked, fused, and individually located payload regions;
- malformed or overlapping payload regions;
- unsupported router/expert/combine operation rejection during compilation;
- fingerprint changes for every execution-affecting field;
- fingerprint stability for observability-only configuration;
- CPU/CUDA lowering from the same neutral program;
- no family identity in backend tests;
- packed and standalone consumption of the same lowered descriptor.

## Expert policy and host cache

- deterministic frequency/recency traces and decay;
- true miss versus coalesced wait;
- zero/undersized budgets and payload larger than budget;
- same-key coalescing and different-key concurrency;
- loader failure/retry and waiter propagation;
- lease blocks eviction and duplicate release is detected;
- queue saturation and shutdown semantics;
- concurrent metrics snapshots.

## CPU backing and pack I/O

- existing pack opens without payload reads;
- index validation materializes no matrix;
- first build writes/releases one expert at a time;
- malformed/truncated entries and dimension mismatches fail;
- independent concurrent reads are safe;
- cloned sessions share backing/cache;
- active objects survive eviction until GEMVs complete;
- native GGUF uses no redundant cache;
- memory is bounded by documented budgets and temporary buffers.

## CUDA expert residency

- fully resident path;
- one and several unique cold experts;
- cold set equal to and greater than capacity;
- probation reservation and protected retention;
- active-batch pinning;
- prefetch cannot evict active slots;
- host lease survives H2D completion;
- transfer failure publishes nothing;
- pointer/slot tables remain consistent;
- concurrent sessions sharing model weights;
- per-device isolation;
- source adapters driven by compiled payload schema;
- zero host allocations and no unconditional synchronization on all-resident decode;
- compact cold transfer;
- decode/prefill and packed/standalone consistency.

## Architecture and dependency boundaries

- catalog duplicate detection/freezing and custom injection;
- custom family modules with/without built-ins;
- fake neutral repository through CPU/CUDA compilers;
- no family dependency in generic runtime;
- no format dependency in generic cache;
- no architecture dispatch in backends;
- no concrete services in C API;
- no family identity required to compile or execute synthetic MoE programs.

## Dense topology, chat, tools, factories, and packed execution

Preserve focused tests for dense topology mapping/fingerprints, exact family prompt fixtures, JSON/tag handling, tool calls/results, backend factory creation/failure, ABI validation, packed input validation, decode/prefill parity, completion boundaries, and zero steady-state allocations.

---

# Performance constraints

## Fully resident steady state

Do not introduce:

- host heap allocation;
- disk access or worker submission;
- full router-selection D2H copies;
- unconditional stream synchronization;
- architecture/checkpoint probing;
- string-based tensor lookup;
- repeated semantic-program or compatibility-key construction;
- per-token plan selection;
- virtual dispatch inside layer/token loops.

## Cold admission

Permit only bounded work required by a real miss:

- compact unique cold list;
- capped parallel I/O;
- preallocated or explicitly bounded metadata;
- no storage I/O under a global lock;
- event-based transfer completion;
- atomic publication after mandatory payloads are valid;
- backpressure instead of unbounded queues;
- explicit failure when the required set cannot fit.

## Backend lowering

All semantic interpretation, capability checks, payload binding, kernel choice, and function selection happen at model compilation/binding time. Decode and prefill consume compact executable records.

Any material regression must be measured, explained, and fixed or explicitly rejected. File count and line count are not success metrics.

---

# CI and verification requirements

Required commands, adapted to the environment:

```text
python scripts/check_architecture_boundaries.py --root .
python scripts/dev.py verify --backend cpu --build-type Release
python scripts/dev.py verify --backend cuda --arch <arch> --build-type Release
ctest --test-dir <configured-build-dir> --output-on-failure
git diff --check
```

Report separately:

1. CPU compile/runtime tests;
2. CUDA compile validation;
3. host-only CUDA-linked tests;
4. CUDA runtime tests on a real GPU;
5. neutral MoE program/lowering tests;
6. numerical parity;
7. allocation and synchronization evidence;
8. hit/miss and packed performance benchmarks.

Hosted CUDA compilation without a GPU does not prove runtime residency, kernel correctness, or performance. Run CUDA-independent semantic/source/cache tests in hosted builds where possible; use a real-device runner or explicit local evidence for GPU claims.

---

# Commit and working rules

- work directly on the current target branch unless instructed otherwise;
- do not open a pull request;
- use small coherent commits describing architectural outcomes;
- keep correctness fixes separate from optional tuning where practical;
- add regression tests before changing confirmed behavior;
- preserve no dead compatibility path without evidence it is required;
- add abstractions only for demonstrated semantic variation or a clear stable boundary;
- do not hide synchronization/allocation inside innocent-looking helpers;
- do not replace explicit code with a parameter/callback maze;
- do not stop after documentation changes;
- do not claim completion because code merely moved files.

---

# Final deliverable

When implementation is complete, provide:

1. resulting commit range;
2. mapping of findings to files and tests;
3. before/after SOLID, DRY, multi-MoE, packed, and residency assessment;
4. the final neutral MoE extension path with one synthetic and one real family example;
5. commands and exact results;
6. semantic-program and lowering coverage;
7. cache-policy/failure-transition results;
8. CPU pack laziness and bounded-memory evidence;
9. CUDA hit/miss allocation, synchronization, and benchmark evidence;
10. CI status separated into compile-only and real-device validation;
11. remaining debt by correctness, architecture, DRY, performance, and optional enhancement;
12. explicit confirmation that generic runtime knows no concrete family/format, backends use no family identity for MoE, C API constructs no concrete backend, packed pipelines own workflows, and expert tiers have explicit ownership/lifetime contracts;
13. updated executable documentation for adding another MoE family.

Do not claim completion for any item lacking source, tests, and—where performance is involved—measurement evidence.
