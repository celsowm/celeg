# Celeg SOLID, DRY, Packed Execution, and Expert Residency Refactoring Plan

**Repository:** `celsowm/celeg`  
**Target branch:** current default branch (`master`)  
**Reviewed head:** `79fdba3347cdec14b23a8efd3eb8166e486fbe6f`  
**Previous plan baseline:** `7d000d1607d7081547f6dfece82f2682b0535ac6`  
**Review date:** 2026-08-04  
**Review delta:** 36 commits after the previous baseline  
**Primary goal:** finish the SOLID and DRY refactoring while preserving Celeg's CPU/CUDA performance model and making the new SSD → RAM → VRAM expert hierarchy correct, bounded, observable, and independently testable.

---

## Role

Act as a senior C++/CUDA inference-runtime architect working directly on the Celeg repository.

This document is an execution plan, not a request for another architecture report. Verify every finding against the current branch, implement the work in coherent stages, add tests before changing behavior for confirmed defects, and leave the repository buildable after every completed stage.

Do not reintroduce architecture enums, model variants, checkpoint-format branching in backends, concrete service construction in generic APIs, or virtual dispatch inside transformer inner loops.

---

## What changed since the previous plan

The earlier SOLID/DRY findings remain largely valid, but the repository now also contains a substantial expert-storage subsystem:

- CUDA SSD backing with a shared host cache and per-layer VRAM residency;
- active-FFN pinning and probation-slot reservation;
- frequency-aware host and device cache policies;
- an asynchronous expert I/O worker pool;
- CPU disk-backed experts loaded lazily from `.lfmpack`;
- concurrent indexed CPU pack reads;
- shared CPU expert caches across cloned sessions;
- streaming pack creation that does not retain all expert payloads in RAM;
- tests proving lazy pack payload access and basic cache behavior;
- documentation for CPU and CUDA SSD tiers.

These are valuable capabilities, but they introduced new ownership, dependency, lifetime, metrics, synchronization, and DRY concerns that the previous plan did not cover.

The implementation order must therefore change: stabilize and decouple expert storage first, then continue with runtime modules, backend factories, text/model DRY, and packed-execution ownership.

---

## Definition of success

The refactoring is complete only when all of the following are true:

1. a new model family can register architecture, chat, tool, tokenizer, vision, and other family-owned providers through one coherent registration boundary;
2. generic runtime composition does not include, construct, or name concrete model families;
3. backend selection returns a fully constructed service bundle through a real backend factory;
4. the C API translates ABI data but does not construct concrete CPU or CUDA services;
5. packed decode and ragged prefill have distinct workflow owners rather than pass-through wrappers around a god executor;
6. SSD expert source access, host caching, GPU residency, transfer scheduling, and usage metrics have explicit owners;
7. a generic runtime cache does not depend on Safetensors, CUDA, or the LFM-specific `gate_up`/`down` payload layout;
8. active expert payloads cannot be evicted until their CPU GEMVs or CUDA FFN transfers and execution have completed;
9. the all-resident CUDA expert path performs no disk I/O, heap allocation, worker-pool submission, or unconditional stream synchronization;
10. cold expert admission is bounded by configured RAM/VRAM budgets and fails before publishing partial residency state;
11. CPU `.lfmpack` opening remains metadata-only and payloads remain lazy until a routed expert is acquired;
12. cache hit, true miss, coalesced wait, eviction, storage-read, host-to-device transfer, and wait-time metrics have documented semantics;
13. shared semantic rules—packed compatibility, common batch validation, cache keys/scoring where truly identical, JSON escaping, tagged-block handling, and dense topology construction—have one authoritative implementation;
14. CPU and CUDA mechanisms remain separate where their storage, synchronization, and kernel behavior genuinely differ;
15. no architecture/checkpoint probing, repeated tensor lookup, plan construction, or dynamic polymorphic graph traversal is introduced into execution hot paths;
16. tests and architecture checks cover every extracted boundary and every failure transition;
17. CI output distinguishes compile-only CUDA validation from runtime validation on a real GPU;
18. documentation describes the final implementation rather than an intended design.

Target assessment after completion:

| Area | Target |
|---|---:|
| Single Responsibility | 8.3+ |
| Open/Closed | 8.5+ |
| Liskov Substitution | 8.5+ |
| Interface Segregation | 8.8+ |
| Dependency Inversion | 8.6+ |
| DRY | 8.4+ |
| Packed execution | 8.2+ |
| Expert residency/storage | 8.3+ |
| Overall architecture | 8.5+ |

---

## Preserve these improvements

Do not undo the following design and performance gains:

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
- CPU cache sharing across sessions cloned from the same compiled model;
- coalescing concurrent loads of the same expert;
- CUDA active-batch pinning;
- CUDA probation-slot reservation for the complete unique cold set;
- device-side cold-set discovery;
- failure before host-visible state commit;
- closed-domain switches that are explicitly documented and exhaustively tested.

Preserve the core execution strategy:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

For disk-backed experts, extend it with:

```text
index once
read on demand
lease while active
publish only after transfer
release after completion
```

---

## DRY policy

DRY means **one authoritative representation of knowledge**, not zero similar lines.

A duplication is actionable when copies encode the same semantic rule and can silently diverge. Current examples include:

- what constitutes a cache key;
- how access heat decays;
- how frequency and recency combine into an eviction score;
- when an entry moves from probation to protected status;
- what a cache hit or miss counter means;
- which options make packed sessions compatible;
- common packed-session validation;
- JSON escaping;
- tagged tool-call block lifecycle;
- standard dense-decoder topology construction;
- model-family registration across catalogs.

Repetition is acceptable when implementations have independent reasons to change. Do not force one generic cache implementation over:

- CPU quantized expert objects;
- raw host payload arenas used for CUDA transfers;
- per-layer VRAM slot and pointer-table management;
- native GGUF mappings;
- BF16, INT8, INT4, GGUF, DP4A, tensor-core, or other materially different kernels;
- CPU mutex/future coordination and CUDA stream/event coordination;
- model-family probe rules and exceptional metadata semantics;
- tensor naming policies for different checkpoint conventions;
- tests whose similarity documents independent contracts;
- intentionally closed public ABI enum translation.

Before extracting shared code, answer:

1. Is this the same knowledge or merely similar syntax?
2. Must all copies change together when the rule changes?
3. Can the contract be named without mentioning unrelated callers?
4. Does the extraction reduce conditionals and parameters rather than move them?
5. Can it be tested directly without constructing a complete model/backend?
6. Does it preserve the hot-path allocation and dispatch model?

If any answer is negative, keep the mechanisms separate and document the intentional difference.

---

# Current high-priority findings

## 1. CUDA expert residency has become a second god orchestrator

### Evidence

`SharedModelWeights::ensure_moe_experts_resident` currently coordinates:

- per-layer locking;
- compute/transfer stream event dependencies;
- cold-set discovery;
- active-transfer lease cleanup;
- cache hit/miss accounting;
- usage-profile mutation;
- asynchronous disk-read submission;
- source selection between sidecar and repository reads;
- host-cache acquisition;
- waiting for I/O futures;
- host-to-device promotion;
- transfer-event creation;
- residency-table publication;
- resident-expert touching;
- speculative prefetch ranking;
- prefetch submission;
- final event publication.

Its signature receives streams, four events, routing buffers, and several reusable host vectors. This is a parameter maze and a clear SRP boundary failure.

### Required direction

Introduce a CUDA expert-residency coordinator with typed inputs and explicit collaborators. The exact names may differ, but the responsibilities should resemble:

```cpp
struct ExpertResidencyRequest {
    int layer = -1;
    int rows = 0;
    int experts_per_token = 0;
    const int* selected_experts = nullptr;
    const float* route_scores = nullptr;
    cudaStream_t compute_stream = nullptr;
};

struct ExpertResidencyWorkspace;
class IExpertSource;
class IHostExpertCache;
class ExpertLayerResidency;
class CudaExpertResidencyCoordinator;
```

- `ExpertLayerResidency` owns VRAM slots, pointer tables, active-slot protection, and device residency metadata.
- `IExpertSource` owns reading one expert payload from sidecar, SafeTensors random access, pack, or another source.
- `IHostExpertCache` owns host-tier admission, coalescing, leasing, and eviction.
- a bounded I/O executor owns worker threads and queue behavior.
- the coordinator owns one residency transaction: discover, acquire, transfer, publish, prefetch, and complete.
- metrics receive explicit events rather than being mutated opportunistically throughout the workflow.

Do not replace the current method with one equally broad helper class.

### Acceptance criteria

- no residency method takes a long list of events and scratch vectors;
- source selection is not embedded in the transfer orchestration;
- VRAM slot code does not read checkpoint formats;
- host-cache code does not publish CUDA pointer tables;
- a failed read or transfer leaves the previous residency map valid;
- all slots used by the active FFN remain protected until the FFN completion event;
- the coordinator is testable with fake source/cache/residency collaborators;
- packed and standalone paths use the same coordinator without duplicating the residency algorithm.

---

## 2. `PinnedExpertCache` is in the wrong abstraction layer

### Evidence

`include/celeg/runtime/cache/pinned_expert_cache.hpp` currently:

- includes `celeg/checkpoint/formats/safetensors.hpp` from a supposedly generic runtime cache;
- hardcodes two payload regions named `gate_up` and `down`;
- contains cache storage, lease behavior, frequency tracking, coalescing, metrics, and `ExpertIoManager` in one header;
- is named “Pinned” even though its default allocator is ordinary `malloc` and pinning is injected by a backend.

The cache is therefore neither format-neutral, model-layout-neutral, nor strictly pinned.

### Required direction

Choose one honest boundary:

1. make it a CUDA-backend-owned host staging cache; or
2. make it a genuinely neutral host blob cache driven by an explicit payload layout and allocator.

A neutral direction may use concepts such as:

```cpp
struct ExpertKey {
    int layer = -1;
    int expert = -1;
    auto operator<=>(const ExpertKey&) const = default;
};

struct ExpertPayloadLayout {
    std::size_t bytes = 0;
    std::span<const PayloadRegion> regions;
};

class ExpertLease {
public:
    std::span<const std::byte> payload() const;
};
```

Keep LFM-specific region interpretation next to the LFM/CUDA binding code. Keep SafeTensors access inside a source adapter. Move the worker pool to an execution/I/O component rather than a cache header.

### Acceptance criteria

- generic runtime headers do not include concrete checkpoint formats;
- the host cache does not name `gate_up`, `down`, LFM, CUDA, or SafeTensors;
- alternatively, a CUDA-specific cache is physically located under the CUDA backend and named accordingly;
- allocator behavior is explicit in the type/configuration;
- I/O worker lifetime is independent of cache storage lifetime;
- focused builds can compile the cache without CUDA or checkpoint-format headers.

---

## 3. Cache policy, capacity, and metrics semantics are duplicated and drifting

### Evidence

Three different caches now encode related policy knowledge:

- `CpuExpertCache` uses decayed heat, a recency bonus, a protected budget, and promotion after repeated access;
- `PinnedExpertCache` keeps a separate heat map and chooses the coldest unleased slot without the same probation/protected model;
- `ExpertLayerCache` combines recency, route scores or observed accesses, protected/probation slots, and batch pinning.

Some differences are intentional because the tiers have different costs. Others are accidental duplication:

- expert key packing is repeated;
- heat-decay constants and formulas are repeated;
- hit/miss definitions differ;
- coalesced host-cache waiters count as misses in one cache but have a separate counter in another;
- a host cache silently creates one slot when the configured budget is smaller than one expert, while the CPU cache declines admission;
- policy names do not fully describe which tier and scoring inputs they control.

### Required direction

Do **not** create a generic `ExpertCache<T, 20 callbacks>`.

Extract only stable policy/value contracts, for example:

- `ExpertKey` and hashing;
- decay calculation;
- an explicit frequency/recency score primitive;
- cache counter definitions;
- budget/admission result types;
- policy configuration validation.

Let each tier retain its own storage, locking, transfer, and eviction mechanism. Document intentional differences in admission and protection.

### Acceptance criteria

- one definition exists for every counter and policy name;
- `true_misses` means a caller became the elected loader;
- `coalesced_waits` is separate from true misses;
- budget smaller than one payload has explicit behavior and a test;
- protected/probation semantics are either shared or explicitly different by tier;
- scoring tests use deterministic access traces;
- CPU, host, and GPU caches expose comparable per-tier metrics without pretending their mechanisms are identical.

---

## 4. The CUDA residency path violates or obscures the hot-path contract

### Evidence

The current residency path includes:

- unconditional `cudaStreamSynchronize` inside device cold-set resolution;
- a full selected-expert D2H copy even though comments describe copying only the compact cold list;
- a second full selected-expert D2H copy on a miss path;
- per-miss `std::vector<ExpertHostLease>` and `std::vector<std::future<void>>` construction;
- synchronous waiting for all submitted disk reads;
- host-side touching and metrics loops over selected experts;
- speculative prefetch work in the same orchestration function.

Cold misses may legitimately block the current token, but the all-resident path must stay much cheaper and its synchronization contract must be explicit.

### Required direction

Define separate performance contracts:

**All-resident hit path**

- no heap allocation;
- no worker-pool submission;
- no disk read;
- no full-selection D2H copy;
- no unconditional stream synchronization;
- only event dependencies required for correctness.

**Cold admission path**

- bounded, preallocated host metadata;
- one compact cold list;
- bounded parallel reads;
- no global lock held during storage I/O;
- event-based host-to-device completion;
- one atomic residency publication after all required transfers are valid.

Move usage statistics and policy touches to device-side compact data or preallocated host buffers where appropriate.

### Acceptance criteria

- tests or instrumentation prove zero host allocations on repeated all-resident decode;
- the all-resident path does not call `cudaStreamSynchronize`;
- the implementation transfers only data required by the current policy;
- miss-path allocations are preallocated or explicitly bounded and measured;
- prefetch cannot delay or invalidate current-token admission;
- before/after hit-path and miss-path timings are recorded.

---

## 5. Expert lease, transfer, publication, and shutdown invariants are implicit

### Evidence

Correctness currently depends on interactions among:

- `ExpertHostLease` reference counts;
- `ResidencyController::inflight_transfers`;
- transfer events;
- `batch_pinned` VRAM slots;
- cache loading flags and generations;
- worker-pool shutdown;
- sidecar/repository lifetimes;
- shared model-weight lifetime.

Some invalid states are hidden rather than rejected. For example, `release_slot` clamps a negative reference count back to zero. Queue depth and very small cache budgets also need explicit validation.

### Required direction

Model the residency transaction and slot lifecycle explicitly. A useful state vocabulary is:

```text
empty -> loading -> host-resident -> transferring -> device-resident
                                      -> failed

device-resident -> active -> device-resident
                -> evicted
```

Use RAII tickets/leases that hold the exact resources required until a completion event is observed. Make state publication transactional.

### Acceptance criteria

- reference-count underflow is impossible or detected in tests;
- duplicate release is not silently accepted;
- queue depth, worker count, cache budget, and payload size are validated;
- executor shutdown either drains or cancels queued work according to a documented rule;
- failures propagate to every coalesced waiter;
- a failed load can be retried cleanly;
- a failed H2D transfer does not publish a device pointer;
- active slots and host payloads remain alive until the corresponding completion event;
- destruction order is explicit and tested.

---

## 6. CPU disk-backed expert ownership is spread across compiled-model internals

### Evidence

CPU expert backing currently spans:

- `configure_cpu_expert_backing`;
- mutable fields on `CpuCompiledModel::Shared`;
- `CpuPackReader`;
- `CpuExpertCache`;
- mutation and clearing of MoE vectors in `weight_store`;
- `Shared::acquire_expert`;
- forward-path knowledge of disk-cached experts.

The mechanism works, but the compiled model remains the owner of source configuration, cache configuration, tensor-name construction, validation, and acquisition.

### Required direction

Introduce a CPU-owned expert backing object with one explicit responsibility: resolve and lease quantized expert weights.

Possible conceptual boundary:

```cpp
class CpuExpertBackingStore {
public:
    std::shared_ptr<const CpuExpertWeights> acquire(ExpertKey key);
    CpuExpertBackingMetrics metrics() const;
};
```

It may own a pack reader, cache, tensor-name resolver, and immutable shape validation. Native GGUF should remain a separate mapped-storage path rather than being forced through this object.

### Acceptance criteria

- model forward code asks for an expert lease without knowing pack-reader details;
- pack entry naming and dimension validation have one owner;
- opening an existing pack does not read expert payloads;
- first-time pack construction releases each expert after writing;
- cloned sessions share the backing store/cache intentionally;
- native GGUF continues to use mapped block storage without a redundant Celeg cache;
- concurrent reads of different experts remain possible.

---

## 7. Packed execution still concentrates too many responsibilities

`PackedDecodeExecutorImpl` still owns or coordinates shared workspace state, compatibility, decode/prefill validation, metadata staging, GEMM setup, embedding, layer traversal, attention, convolution, dense FFN, MoE, completion, host commit, and metrics.

`PackedDecodePipeline` and `PackedPrefillPipeline` remain mostly pass-through wrappers. Moving declarations without moving workflow authority does not satisfy SRP.

### Required direction

- make decode orchestration owned by `PackedDecodePipeline`;
- make ragged-prefill orchestration owned by `PackedPrefillPipeline`;
- keep preallocated shared resources in a clearly named shared workspace/resource object;
- use the same CUDA expert-residency coordinator as standalone execution;
- give validation, metadata staging, completion/commit, and metrics narrow owners;
- preserve direct, prebound hot-path calls.

### Acceptance criteria

- pipeline `run()` methods contain operation-specific sequences;
- shared state does not know both complete algorithms;
- expert residency is not duplicated between packed and standalone paths;
- failed CUDA work cannot mutate host-visible session state;
- packed/non-packed and ragged/standard parity tests remain green;
- steady-state allocation tests remain green.

---

## 8. `IBackendFactory` is still not a real factory

The current abstraction identifies/supports backends, while generic composition still constructs concrete CPU/CUDA services.

### Required direction

Define a neutral backend creation request and a factory result exposing serving roles:

```cpp
struct BackendCreationRequest {
    std::filesystem::path model_path;
    std::size_t maximum_context = 0;
    BackendConfiguration configuration;
    std::shared_ptr<const RuntimeContext> runtime;
};

class IBackendFactory {
public:
    virtual ~IBackendFactory() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(BackendKind backend) const = 0;
    virtual std::unique_ptr<serve::ServiceBundle> create(
        BackendCreationRequest request) const = 0;
};
```

The exact shape may differ, but the selected factory must own concrete construction.

### Acceptance criteria

- `celeg_engine_create` does not include or instantiate concrete inference services;
- adding a backend does not add another C API construction branch;
- unsupported builds fail explicitly before partial mutation;
- public ABI mappings remain outside backend code;
- backend creation diagnostics remain useful;
- CPU-only builds do not reference CUDA implementation symbols.

---

## 9. Generic runtime composition still knows concrete model families

Generic runtime composition still includes concrete family knowledge, including Gemma4 vision construction.

### Required direction

Introduce a coherent family module boundary:

```cpp
class IRuntimeModule {
public:
    virtual ~IRuntimeModule() = default;
    virtual std::string_view id() const = 0;
    virtual void register_into(RuntimeBuilder& builder) const = 0;
};
```

A family module may register architecture, chat profile, tool codec, tokenizer provider, vision provider, and other cold-path family-owned services. Static built-in registration is sufficient; dynamic plugins are not required.

### Acceptance criteria

- each family has one registration entry point;
- generic `src/runtime/**` contains no family includes, strings, types, or constructors;
- adding a vision-capable family does not edit generic runtime composition;
- duplicate registrations are rejected;
- catalogs freeze after construction;
- custom runtimes can inject modules with or without built-ins.

---

## 10. Family registration knowledge remains duplicated

The supported-family list is repeated across architecture registration, chat profiles, tool codecs, tokenizer behavior, vision bootstrap, source lists, and tests.

### Required direction

Use family modules to make runtime contributions atomic. Do not create a central mega-structure with nullable fields for every optional capability.

### Acceptance criteria

- adding a family requires one built-in module registration plus family-owned code;
- families without tools or vision do not implement fake methods;
- family module tests verify expected contributions;
- generic catalogs remain independently testable;
- build/source registration is generated or guarded against partial family registration where practical.

---

## 11. Dense decoder topology construction is duplicated

MiniCPM5 and SmolLM3 repeat standard dense topology initialization and paired GGUF/JSON metadata access. Probe rules and exceptional defaults remain family-specific, but standard topology invariants are shared knowledge.

### Required direction

Extract a neutral dense-decoder topology builder driven by explicit metadata mappings and defaults. Avoid family names and boolean switches in the neutral builder.

### Acceptance criteria

- standard dense topology invariants have one implementation;
- family files retain probe identity and genuine exceptions;
- GGUF and SafeTensors/JSON paths have focused builder tests;
- topology fingerprints remain stable unless fixing a demonstrated bug;
- MiniCPM5 and SmolLM3 resolution tests remain independent.

---

## 12. JSON and tagged tool-call parsing have multiple implementations

JSON escaping, trimming, object-member scanning, delimiter matching, tagged-block removal, call-ID generation, and incomplete-block handling appear in multiple template/codec implementations.

### Required direction

Extract small protocol-neutral utilities for JSON string escaping and tagged-block lifecycle. Use the project's real JSON parser for arbitrary JSON rather than maintaining another incomplete parser.

### Acceptance criteria

- JSON escaping has one authoritative implementation;
- Unicode, control characters, quotes, slashes, and empty strings are covered;
- shared tagged-block scanning/removal has one implementation;
- protocol-specific syntax remains in each codec;
- malformed and incomplete output is tested per family;
- mixed assistant text and parallel calls are preserved.

---

## 13. Chat templates and tool codecs overlap in responsibility

Central chat-template code still contains family-specific formatting and constructs tool-definition syntax that overlaps with family-owned codecs.

### Required direction

- move family templates under family modules;
- keep neutral chat contracts/catalogs under generic text code;
- give tool syntax one owner, preferably the native codec where one exists;
- keep the conversation skeleton responsible for role sequencing and message placement.

### Acceptance criteria

- generic chat code contains no Granite, Gemma4, LFM2, MiniCPM5, or SmolLM3 rules;
- tool syntax is not reconstructed independently in template and codec;
- prompt fixtures remain byte-identical unless a documented bug is fixed;
- rendering and parsing tests remain family-specific.

---

## 14. Packed common batch validation is duplicated

Decode and ragged prefill repeat empty-batch, capacity, owner, duplicate-session, plan, device, eligibility, storage, and option checks.

### Required direction

Extract common validation and layer operation-specific validation on top:

```cpp
const PackedSessionContext& validate_common_batch(
    std::span<const PackedSessionContext> sessions,
    PackedOperation operation) const;
```

### Acceptance criteria

- common invariants have one implementation;
- errors retain operation context;
- decode does not depend on prefill-only data;
- duplicate detection is efficient;
- malformed input fails before CUDA work or host-state mutation.

---

## 15. Packed compatibility knowledge is duplicated

Compatibility-affecting options are manually compared while related knowledge also exists in execution plans and fingerprints.

### Required direction

Create an immutable compatibility identity produced during compilation:

```cpp
struct PackedCompatibilityKey {
    WeightMode weight_mode;
    KvCacheMode kv_cache_mode;
    GemmBackend gemm_backend;
    AttentionMode attention_mode;
    auto operator<=>(const PackedCompatibilityKey&) const = default;
};
```

Determine fields from execution semantics, including expert-storage/residency compatibility where shared packed execution depends on it.

### Acceptance criteria

- one function defines the key;
- the key is part of or derived from the immutable execution plan;
- resource/device/source ownership identities are compared explicitly;
- every compatibility-affecting option has a test;
- observability-only options do not create false incompatibility;
- fingerprints and equality cannot drift independently.

---

## 16. C API CPU/CUDA composition remains duplicated

The C API repeats backend selection, option conversion, concrete service creation, and bundle wrapping. Local helpers would hide the duplication rather than fix dependency direction.

### Required direction

Fix backend factories first. Keep ABI functions focused on public struct-size checks, enum validation, and translation into neutral configuration.

### Acceptance criteria

- C API entry points are small and backend-neutral;
- public validation is centralized;
- factories are independently testable;
- invalid ABI input is distinguishable from backend-creation failure;
- CPU-only linkage remains CUDA-free.

---

# SOLID-specific requirements

## Single Responsibility Principle

A valid extraction must move at least one of:

- ownership;
- lifetime;
- dependency direction;
- invariant enforcement;
- state-transition authority;
- synchronization authority;
- compilation boundary;
- testable contract.

Moving methods into another `.cpp` while the original object still owns all state and decisions is not sufficient.

Prioritize:

- CUDA expert-residency orchestration;
- host expert cache and I/O execution;
- CPU expert backing;
- `PackedDecodeExecutorImpl`;
- generic runtime bootstrap;
- family resolvers/templates;
- C API engine construction.

## Open/Closed Principle

Adding a model family must not require edits across generic runtime, text, backend, and serving layers. Adding a backend must not require another C API branch. Adding an expert source should not require rewriting cache eviction or CUDA slot management.

Closed domains are allowed when explicit and tested. Adding another `switch` case is not automatically OCP compliance.

## Liskov Substitution Principle

- optional behavior belongs in separate capability interfaces;
- a source claiming random access must honor concurrent-read and lifetime contracts;
- cache implementations must honor the same lease and failure semantics they advertise;
- a backend factory claiming support must construct a usable service or fail before partial mutation;
- asynchronous completion contracts must define when payloads, pointer tables, and host-visible state become observable;
- borrowed pointers must document their owner and completion lifetime;
- family modules must work in custom builders without hidden globals.

## Interface Segregation Principle

Do not replace narrow repository, serving, source, cache, or residency roles with a new composite god interface.

A family without vision/tools, a source without random access, or a cache without asynchronous loading must not implement placeholder methods.

## Dependency Inversion Principle

Generic layers depend on neutral contracts. Concrete families, formats, storage sources, CPU/CUDA mechanisms, and services point inward through registration, compilation, adapters, and factories.

Forbidden directions include:

```text
src/runtime/**                 -> celeg/models/**
src/runtime/cache/**           -> checkpoint/formats/**
include/celeg/model/**         -> celeg/backend/cuda/**
src/backend/**                 -> architecture identity dispatch
src/backend/**                 -> concrete checkpoint-format dispatch
VRAM residency                -> SafeTensors/sidecar parsing
host cache                    -> CUDA pointer-table publication
src/api/**                    -> concrete inference service construction
```

---

# Implementation stages

Complete stages in order unless current code evidence proves a dependency requires a different sequence. Keep every stage independently buildable and testable.

## Stage 0 — Re-baseline and protect behavior

1. Record the current head and changes since `79fdba3347cdec14b23a8efd3eb8166e486fbe6f`.
2. Run architecture boundary checks.
3. Build CPU and CUDA targets available in the environment.
4. Run CPU tests and host-only CUDA tests that do not require a device.
5. Record which CUDA tests were compiled, which ran without a device, and which ran on a real GPU.
6. Capture prompt fixtures for all families.
7. Capture packed/non-packed and standard/ragged parity where hardware exists.
8. Capture CPU expert pack, CPU cache, CUDA cache-hit, CUDA cache-miss, and fully resident performance baselines.
9. Record host/device allocations and synchronization points for expert decode and prefill.

Do not claim CUDA runtime validation for compile-only jobs.

## Stage 1 — Lock down expert-storage semantics

Before structural changes, add focused tests for:

- exact cache metric definitions;
- budget smaller than one expert;
- zero budget;
- duplicate release/reference underflow;
- worker count and queue-depth validation;
- loader failure and retry;
- failure propagation to all waiters;
- active lease preventing eviction;
- active FFN preventing VRAM eviction;
- unique cold set equal to and greater than VRAM capacity;
- transactional pointer-table publication;
- native GGUF bypass;
- pack opening without payload reads;
- cloned CPU sessions sharing one cache;
- concurrent different-expert reads.

Fix confirmed correctness defects before broad refactoring.

## Stage 2 — Extract neutral policy and metrics values

Extract only stable semantic primitives:

- expert key/hash;
- decay/scoring math where identical;
- counter definitions;
- budget/admission result;
- policy configuration validation.

Keep CPU, host-staging, and VRAM caches separate.

## Stage 3 — Separate expert source, host cache, and I/O executor

- remove concrete checkpoint-format includes from generic cache code;
- move LFM payload layout interpretation out of the cache;
- give sidecar and random-access repository reads source adapters;
- separate the bounded worker pool from cache storage;
- make allocator/pinned-memory behavior explicit;
- add fake-source and deterministic executor tests.

## Stage 4 — Refactor CUDA residency orchestration

- introduce a typed request and preallocated workspace;
- separate per-layer VRAM slot state from transaction orchestration;
- centralize transfer lease/event lifetime;
- make publication transactional;
- remove redundant full-selection D2H copies;
- eliminate unconditional synchronization from the all-resident path;
- keep prefetch subordinate to current-token correctness;
- use the same coordinator from standalone and packed execution.

## Stage 5 — Encapsulate CPU expert backing

- introduce a CPU-owned backing store/source boundary;
- centralize pack naming and dimension validation;
- preserve lazy payload reads and streaming pack creation;
- keep native GGUF mapped storage separate;
- preserve shared cache ownership across cloned sessions.

## Stage 6 — Shared low-risk DRY utilities

Implement and test:

- JSON escaping;
- tagged-block scanning/removal;
- common metadata helpers only where semantics are identical;
- common packed batch validation;
- immutable packed compatibility identity.

## Stage 7 — Dense decoder topology builder

Extract standard dense topology invariants from MiniCPM5 and SmolLM3 while keeping probe identity and exceptional policy local.

Do not force LFM2, Granite, or Gemma4 through the builder unless they fit without family flags.

## Stage 8 — Family-owned chat/template/tool modules

Move family templates and codecs into family directories. Remove duplicate tool-rendering ownership. Preserve exact prompt fixtures.

## Stage 9 — Runtime family modules

Introduce the family registration boundary and remove concrete family knowledge from generic runtime bootstrap.

## Stage 10 — Real backend factories and neutral C API composition

Make factories create service bundles. Remove concrete CPU/CUDA service construction from the C API and generic composition roots.

## Stage 11 — Packed execution ownership

Move operation-specific orchestration into real decode and prefill pipeline owners while retaining explicit preallocated shared resources and the common expert-residency coordinator.

## Stage 12 — Expand architecture, DRY, and hot-path guards

Extend `scripts/check_architecture_boundaries.py` or add focused companion checks for stable rules:

- generic runtime includes of `celeg/models/`;
- generic runtime cache includes of concrete checkpoint formats;
- concrete backend service construction from `src/api`;
- family names/types in generic runtime or generic text registration;
- direct architecture/checkpoint-format dispatch in backends;
- plan construction or direct allocation in packed steady-state paths;
- pass-through packed pipelines;
- direct storage reads from VRAM residency code;
- obsolete compatibility paths.

Do not create fragile regex checks for ordinary local implementation details.

## Stage 13 — Full verification and documentation

Run the complete verification matrix, publish benchmark/allocation evidence, update `docs/EXPERT_STORAGE.md` and `docs/PACKED_EXECUTION.md`, and replace plan statements with links to final contracts and tests.

---

# Testing requirements

## Expert policy and host cache

- deterministic frequency/recency traces;
- decay across multiple intervals;
- protected/probation transitions where applicable;
- true miss versus coalesced wait accounting;
- zero and undersized budgets;
- one payload larger than the entire budget;
- multiple payload sizes if supported;
- same-key concurrent coalescing;
- different-key concurrent loading;
- loader failure and retry;
- failure propagation to every waiter;
- lease blocks eviction;
- duplicate release detection;
- executor queue saturation;
- executor shutdown with queued and active work;
- metric snapshots under concurrency.

## CPU expert backing and pack I/O

- existing pack opens without payload reads;
- index validation does not materialize matrices;
- first pack build writes/releases one expert at a time;
- temporary pack cleanup on failure;
- concurrent indexed reads use independent read positions safely;
- malformed/truncated entry rejection;
- dimension mismatch rejection;
- cloned sessions share cache/backing ownership;
- active expert objects survive cache eviction until GEMVs complete;
- native GGUF performs no redundant pack/cache path;
- memory stays bounded by cache budget plus documented temporary buffers.

## CUDA expert residency

- fully resident hit path;
- one cold expert;
- several unique cold experts;
- unique cold set equal to capacity;
- unique cold set greater than capacity;
- probation reservation;
- protected hot expert retention;
- active-batch pinning;
- prefetch cannot evict current-batch slots;
- host lease remains alive through H2D completion;
- transfer failure does not publish pointers;
- pointer-table and slot-table publication remains consistent;
- concurrent sessions sharing model weights;
- per-device isolation/fingerprinting;
- sidecar and random-access source adapters;
- all-resident path has zero host allocations;
- all-resident path has no unconditional stream synchronization;
- cold-list transfer is compact;
- decode and prefill use the same residency semantics;
- packed and standalone execution remain consistent.

## Architecture and dependency boundaries

- architecture catalog duplicate detection and freezing;
- custom architecture injection;
- custom runtime modules with and without built-ins;
- fake neutral repository compiling through CPU and CUDA compilers;
- no family dependency in generic runtime;
- no concrete format dependency in generic cache code;
- no architecture/checkpoint dispatch in backends;
- no concrete services in C API composition.

## Dense topology

- GGUF and SafeTensors/JSON key mapping;
- token defaults/overrides and multiple EOS IDs;
- attention slot/layer mapping;
- per-layer positional-encoding exceptions;
- topology validation and fingerprint stability;
- independent MiniCPM5 and SmolLM3 resolution.

## Chat and tools

- exact prompt fixtures for every family;
- JSON escaping of quotes, slashes, controls, UTF-8, and empty strings;
- zero, one, and parallel tool calls;
- assistant text before, between, and after calls;
- complete, incomplete, and malformed tagged blocks;
- tool results and role capability enforcement;
- no duplicate tool-definition rendering.

## Backend factories and C API

- CPU factory creation;
- CUDA factory creation when compiled;
- unsupported CUDA build behavior;
- invalid backend and option mapping;
- factory failure diagnostics;
- C API smoke tests without concrete-service dependencies;
- CPU-only shared libraries remain CUDA-free.

## Packed execution

- empty, oversized, null, duplicate, wrong-device, wrong-plan, wrong-storage, and incompatible-option batches;
- common decode/prefill validation consistency;
- packed/non-packed decode parity;
- ragged/standard prefill parity;
- mixed finalize and zero-finalize behavior;
- non-equal query width;
- variable per-layer FFN width;
- metadata replacement/rebinding;
- failure before host commit;
- completion and metrics boundaries;
- zero steady-state host/device allocations;
- representative performance thresholds.

---

# Performance constraints

## Fully resident steady state

Do not introduce:

- host heap allocation;
- disk access;
- worker-pool submission;
- full router-selection D2H copies;
- unconditional stream synchronization;
- architecture/checkpoint probing;
- string-based tensor lookup;
- repeated compatibility-key construction;
- per-token plan creation;
- virtual dispatch inside layer/token loops.

## Cold admission

Permit only bounded work required by an actual miss:

- a compact unique cold list;
- capped parallel I/O;
- preallocated or explicitly bounded metadata;
- no storage I/O while holding a global cache/residency lock;
- event-based transfer completion;
- atomic publication after all mandatory payloads are valid;
- backpressure rather than unbounded task queues;
- explicit error if a batch cannot fit the configured cache.

## CPU disk-backed execution

Preserve:

- metadata-only pack opening;
- lazy payload reads;
- concurrent independent reads;
- one-expert-at-a-time pack construction;
- bounded steady-state RAM;
- leases that outlive active GEMVs;
- native GGUF mmap/page-cache behavior.

Any material regression must be measured, explained, and fixed or explicitly rejected. File count and line count are not success metrics.

---

# CI and verification requirements

The current GitHub workflow has important limits:

- Linux CPU runs full `verify` and VNNI/linkage checks;
- Windows CPU runs full `verify`;
- Linux CUDA builds in a CUDA container without a GPU and currently invokes verification with Celeg tests disabled;
- Windows CUDA is compile-only with tests disabled;
- neither hosted CUDA job proves runtime expert residency, kernel correctness, or performance on a real device.

Required repository commands, adapted to the environment:

```text
python scripts/check_architecture_boundaries.py --root .
python scripts/dev.py verify --backend cpu --build-type Release
python scripts/dev.py verify --backend cuda --arch <arch> --build-type Release
ctest --test-dir <configured-build-dir> --output-on-failure
git diff --check
```

Verification reporting must separate:

1. CPU compile and runtime tests;
2. CUDA compile validation;
3. host-only CUDA-linked tests;
4. CUDA runtime tests on a real GPU;
5. numerical parity;
6. allocation/synchronization evidence;
7. performance benchmarks.

Improve CI where practical so CUDA-independent cache/source/policy tests run in CUDA builds even without a device. Real-GPU tests require a self-hosted runner or explicit local evidence; do not pretend a hosted compile job supplies that evidence.

---

# Commit and working rules

- work on the current target branch unless explicitly instructed otherwise;
- do not open a pull request for this plan;
- use small coherent commits describing architectural outcomes;
- keep correctness fixes separate from optional performance tuning where possible;
- add regression tests before changing behavior for confirmed defects;
- do not preserve dead compatibility paths without evidence they are public and required;
- do not add speculative abstractions for hypothetical families, sources, or backends;
- do not hide synchronization or allocation inside innocent-looking helpers;
- do not replace explicit code with a generic parameter/callback maze;
- do not stop after documentation changes;
- do not declare a phase complete merely because code moved files.

---

# Final deliverable

When implementation is complete, provide:

1. the resulting commit range;
2. a mapping of each finding to changed files and tests;
3. before/after SOLID, DRY, packed-execution, and expert-residency assessment with evidence;
4. commands run and exact results;
5. cache-policy and failure-transition test results;
6. CPU pack laziness and bounded-memory evidence;
7. CUDA hit/miss allocation, synchronization, and benchmark evidence;
8. CI status separated into compile-only and real-device validation;
9. remaining debt separated into correctness, architecture, DRY, performance, and optional enhancement categories;
10. explicit confirmation that generic runtime no longer knows concrete families or formats, C API no longer constructs concrete backends, packed pipelines own their workflows, and expert storage tiers have explicit ownership/lifetime contracts;
11. updated documentation pointing to final contracts and executable tests.

Do not claim completion for any item lacking source, test, and—where performance is involved—measurement evidence.
