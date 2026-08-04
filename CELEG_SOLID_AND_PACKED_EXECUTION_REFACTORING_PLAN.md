# Celeg SOLID and DRY Refactoring Execution Prompt

**Repository:** `celsowm/celeg`  
**Target branch:** current default branch (`master`)  
**Reviewed baseline:** `7d000d1607d7081547f6dfece82f2682b0535ac6`  
**Review date:** 2026-08-04  
**Primary goal:** raise Celeg's maintainability from a good but incomplete SOLID/DRY design to a consistently extensible architecture without regressing correctness, CUDA performance, CPU performance, supported model behavior, or binary/API compatibility.

---

## Role

Act as a senior C++/CUDA runtime architect working directly on the Celeg repository.

Do not merely produce another architectural report. Inspect the current code, verify that each finding still exists, implement the refactoring in small coherent stages, update tests and architecture checks, and leave the repository buildable after every completed stage.

The source may have changed after the reviewed baseline. Treat the findings below as hypotheses that must be verified against the current branch before editing. Do not reintroduce already removed interfaces, compatibility paths, or architecture-specific dispatch inside backend hot paths.

---

## Definition of success

The work is complete only when:

1. a new model family can register its architecture, chat profile, tool-call codec, tokenizer or vision providers through one coherent family-owned registration boundary;
2. generic runtime composition does not include or name concrete model families;
3. backend selection creates a backend through a real factory rather than consulting a catalog and then constructing concrete services manually;
4. packed decode and ragged prefill have distinct owners rather than pass-through pipeline wrappers around a god executor;
5. compatibility, metadata parsing, JSON escaping, tagged-block parsing, and common batch validation each have one authoritative implementation where their semantics are genuinely shared;
6. CPU and CUDA remain separate implementations where their mechanisms differ;
7. no steady-state CUDA allocation or architecture/checkpoint probing is introduced into execution hot paths;
8. all existing boundary checks and tests pass, with new regression coverage for every extracted semantic rule;
9. documentation describes the resulting extension path using the actual final code, not a planned design;
10. SOLID and DRY improve without replacing explicit, understandable code with a generic parameter maze.

Target assessment after completion:

| Area | Target |
|---|---:|
| Single Responsibility | 8.0+ |
| Open/Closed | 8.3+ |
| Liskov Substitution | 8.3+ |
| Interface Segregation | 8.7+ |
| Dependency Inversion | 8.5+ |
| DRY | 8.3+ |
| Overall architecture | 8.4+ |

---

## Preserve these improvements

The following current design decisions are valuable and must not be undone:

- `IArchitecture`, `ArchitectureCatalog`, architecture probing, and backend-neutral `ResolvedModel` resolution;
- neutral checkpoint tensor and repository contracts;
- segregated optional repository capabilities such as location, random access, native block storage, and tokenizer data;
- family-owned tensor naming policies;
- explicit token and numerical policies;
- immutable compiled model programs and execution plans;
- CPU and CUDA compiler boundaries;
- precompiled or prebound hot-path decisions;
- `IRequestService`, `ISchedulerDriver`, and `IServiceDiagnostics` as separate serving roles;
- host commit occurring only after successful CUDA completion;
- packed steady-state preallocation and zero-allocation expectations;
- architecture boundary automation;
- closed-domain switches that are explicitly documented and exhaustively tested;
- the execution strategy:

```text
resolve once
compile once
bind once
allocate once
execute many times
```

Do not replace these boundaries with architecture enums, model variants, backend-specific fields in neutral contracts, concrete repository casts, or virtual dispatch inside transformer inner loops.

---

## DRY policy

DRY means **one authoritative representation of knowledge**, not zero similar lines.

A duplication is actionable when the same semantic rule appears in multiple places and can silently diverge. Examples include:

- which options make packed sessions compatible;
- how JSON strings are escaped;
- how tagged tool-call blocks are detected and removed;
- how a standard dense decoder topology is initialized;
- how a model family is registered across architecture, chat, tools, tokenizer, and vision catalogs;
- how common packed-session batch invariants are validated.

Repetition is acceptable when implementations have independent reasons to change. Do not forcibly unify:

- CPU and CUDA execution mechanisms;
- BF16, INT8, INT4, native GGUF, DP4A, tensor-core, or other materially different kernels;
- model-family probe rules and exceptional metadata semantics;
- tensor naming policies for different checkpoint conventions;
- tests whose similarity documents independent contracts;
- C API enum translation switches for intentionally closed public ABI domains;
- small local code fragments when a shared abstraction would require many flags, callbacks, nullable fields, or architecture checks.

Before extracting shared code, answer:

1. Is this the same knowledge or only similar syntax?
2. Must all copies change together when the rule changes?
3. Can the extracted contract be named without referring to two unrelated callers?
4. Does the extraction reduce conditionals and parameters rather than move them?
5. Can the contract be tested directly?

If any answer is negative, keep the implementations separate unless stronger evidence appears.

---

# Current high-priority findings

## 1. Packed execution still concentrates too many responsibilities

`PackedDecodeExecutorImpl` still coordinates or owns most of the following:

- shared workspace state;
- session compatibility;
- decode and prefill batch validation;
- metadata staging and cache invalidation;
- GEMM runtime setup;
- embedding launch;
- transformer-layer traversal;
- attention, convolution, dense FFN, and MoE orchestration;
- decode orchestration;
- ragged-prefill orchestration;
- CUDA completion and synchronization;
- host commit sequencing;
- metrics.

`PackedDecodePipeline` and `PackedPrefillPipeline` currently act mainly as pass-through wrappers that call methods back on the executor state. Moving declarations without moving ownership does not satisfy SRP.

### Required direction

- Make decode orchestration owned by `PackedDecodePipeline`.
- Make ragged-prefill orchestration owned by `PackedPrefillPipeline`.
- Keep shared, preallocated execution state in a clearly named shared state/resource object.
- Give validation, compatibility, metadata staging, completion/commit, and metrics explicit narrow collaborators where useful.
- Avoid introducing virtual calls or dynamic allocation per layer, row, token, or step.
- Preserve one-time plan compilation and workspace allocation.

### Acceptance criteria

- pipeline `run()` methods contain the operation-specific sequence rather than forwarding to executor methods;
- the shared state does not know the entire decode and prefill algorithms;
- common mechanics are reused without merging the distinct operation workflows;
- failed CUDA work cannot mutate host-visible session state;
- packed vs non-packed and ragged vs standard parity tests remain green;
- steady-state allocation tests remain green.

---

## 2. `IBackendFactory` is not yet a factory

The current abstraction exposes identification/support checks, while C API composition still branches on CPU/CUDA and directly constructs `CpuInferenceService` or `CudaInferenceService`. Consulting the catalog and ignoring the selected object is not dependency inversion.

### Required direction

Define a backend creation request that contains only neutral creation inputs and a factory result exposing the serving roles required by callers.

A possible shape is:

```cpp
struct BackendCreationRequest {
    std::filesystem::path model_path;
    std::size_t maximum_context = 0;
    BackendConfiguration configuration;
    VisualEmbeddingProvider visual_embeddings;
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

The exact types may differ, but ownership and dependency direction must be real.

### Acceptance criteria

- `celeg_engine_create` does not include or instantiate concrete CPU/CUDA inference services;
- adding a backend implementation does not require adding another construction branch to the C API;
- unsupported-build behavior remains explicit and testable;
- public ABI mappings remain outside backend implementation code;
- creation errors retain useful diagnostics.

---

## 3. Generic runtime composition still knows concrete families

Generic runtime composition currently includes and constructs Gemma4 vision support directly. This is a concrete family dependency inside a supposedly neutral composition layer.

### Required direction

Introduce a coherent family registration/module boundary. A family module may register:

- architecture resolver;
- chat profile and role capabilities;
- tool-call codec;
- tokenizer provider, when family-specific;
- vision provider, when applicable;
- other cold-path family-owned providers.

Possible direction:

```cpp
class IRuntimeModule {
public:
    virtual ~IRuntimeModule() = default;
    virtual std::string_view id() const = 0;
    virtual void register_into(RuntimeBuilder& builder) const = 0;
};
```

Static built-in registration is acceptable. Dynamic plugin loading is not required. The important rule is that the generic runtime must not include `celeg/models/<family>/...` or name a concrete family.

### Acceptance criteria

- each built-in family has one coherent registration entry point;
- generic `src/runtime/**` contains no model-family includes, strings, types, or constructors;
- adding a vision-capable family does not require editing generic runtime composition;
- duplicate registration remains rejected;
- catalogs are frozen after construction;
- tests prove custom modules can be injected without built-ins.

---

## 4. Family registration knowledge is duplicated

The list of supported families is repeated across architecture registration, chat-profile registration, tool-call codec construction, and vision/provider bootstrap. This duplicates the definition of what belongs to one model family.

### Required direction

Use the family module/bundle to make family composition atomic. Do not create a central mega-structure containing every possible family option as nullable fields. Prefer a narrow registration method on `RuntimeBuilder`.

### Acceptance criteria

- adding a family requires one built-in module registration and family-owned code;
- chat capabilities remain explicit;
- a family without tools or vision does not implement fake optional methods;
- family module tests verify all expected contributions;
- generic catalogs remain usable independently in focused tests.

---

## 5. Dense decoder topology construction is duplicated

MiniCPM5 and SmolLM3 repeat substantial topology initialization and metadata helpers, including standard dense-attention layer arrays, attention slot mapping, FFN arrays, token-policy handling, and JSON/GGUF paired key access.

This is actionable where the same dense-decoder invariant is repeated. Family-specific probing, dimensions used as fallback identity, EOS exceptions, no-RoPE behavior, and model-specific defaults must remain family-owned.

### Required direction

Extract a neutral dense-decoder topology builder driven by explicit metadata mappings and defaults, with small, named override hooks only for real variations.

Possible shape:

```cpp
struct DenseDecoderMetadataKeys {
    std::string gguf_namespace;
    std::string_view hidden;
    std::string_view intermediate;
    std::string_view layer_count;
    std::string_view query_heads;
    std::string_view key_value_heads;
    std::string_view context_length;
};

struct DenseDecoderDefaults {
    int bos_token_id = -1;
    int pad_token_id = -1;
    float norm_epsilon = 0.0f;
    double rope_theta = 0.0;
    ActivationKind activation = ActivationKind::SwiGLU;
};
```

Do not hardcode MiniCPM5 or SmolLM3 names inside the neutral builder.

### Acceptance criteria

- standard dense topology invariants have one implementation;
- family files retain probe logic and genuine exceptional policy;
- the builder does not accumulate boolean switches for every family;
- existing topology fingerprints and resolved graph behavior remain stable unless a prior bug is demonstrated;
- focused builder tests cover GGUF and SafeTensors/JSON metadata paths;
- MiniCPM5 and SmolLM3 resolution tests remain independent.

---

## 6. JSON and tagged tool-call parsing have multiple implementations

JSON string escaping, trimming, object-member scanning, matching delimiters, tagged-block removal, call ID generation, and complete/incomplete detection appear in multiple chat-template and tool-codec implementations.

Some implementations differ subtly, creating a risk that escaping or malformed-generation behavior diverges by family for accidental rather than protocol-specific reasons.

### Required direction

Extract small, protocol-neutral text utilities such as:

```cpp
std::string escape_json_string(std::string_view value);

struct TaggedBlock {
    std::size_t begin = 0;
    std::size_t body_begin = 0;
    std::size_t body_end = 0;
    std::size_t end = 0;
};

std::vector<TaggedBlock> find_tagged_blocks(
    std::string_view text,
    std::string_view opening,
    std::string_view closing);

std::string remove_tagged_blocks(
    std::string_view text,
    std::string_view opening,
    std::string_view closing);
```

Use a real JSON parser already present in the project when parsing arbitrary JSON would otherwise require maintaining another incomplete parser. Do not build a large general JSON framework for a tiny controlled operation.

### Acceptance criteria

- JSON escaping has one authoritative implementation and comprehensive Unicode/control-character tests;
- shared tagged-block lifecycle behavior has one implementation;
- protocol-specific syntax remains inside each codec;
- malformed and incomplete outputs are tested per codec;
- parallel calls and mixed assistant text are preserved;
- no codec silently accepts a format another codec owns.

---

## 7. Chat templates and tool codecs overlap in responsibility

`chat_template.cpp` contains family-specific templates and also constructs tool-definition text that overlaps with family-owned tool-call codecs. This creates two sources of truth for parts of the same protocol and keeps many families in a central translation unit.

### Required direction

- Move family-specific chat-template implementations under their family modules.
- Keep the generic chat profile catalog and neutral interfaces under `src/text` / `include/celeg/text`.
- Define clearly whether tool definition/call/result rendering belongs to the template or codec, then enforce one owner.
- Prefer the codec as the owner of tool syntax when a native codec exists.
- Keep the conversation skeleton/template responsible for role sequencing and message placement.

### Acceptance criteria

- generic chat code does not contain Granite, Gemma4, LFM2, MiniCPM5, or SmolLM3 formatting rules;
- each family owns its template and codec registration;
- tool syntax is not independently reconstructed in two components;
- rendered prompts remain byte-identical for supported fixtures unless a bug fix is documented;
- tool parsing and chat rendering tests remain family-specific.

---

## 8. Packed common batch validation is duplicated

Decode and ragged-prefill validation repeat common invariants such as empty batches, capacity, null owners, duplicate sessions, reference-plan compatibility, device identity, eligibility, and option compatibility.

### Required direction

Extract a common batch validation operation, then layer operation-specific validation on top:

```cpp
const PackedSessionContext& validate_common_batch(
    std::span<const PackedSessionContext> sessions,
    PackedOperation operation) const;
```

Keep prefill-specific row, token, context, and page-table validation separate.

### Acceptance criteria

- common invariants have one implementation;
- error messages retain enough operation context;
- decode does not depend on prefill-only data;
- operation-specific eligibility remains explicit;
- duplicate-session detection is efficient and tested;
- malformed input fails before CUDA work or host-state mutation.

---

## 9. Packed compatibility knowledge is duplicated

The fields that define whether sessions can share an executor are manually compared in `options_compatible`, while related knowledge also exists in execution-plan compilation and fingerprints. A new option can be added to one place and forgotten in another.

### Required direction

Create an explicit immutable compatibility identity produced during compilation, for example:

```cpp
struct PackedCompatibilityKey {
    WeightMode weight_mode;
    KvCacheMode kv_cache_mode;
    GemmBackend gemm_backend;
    AttentionMode attention_mode;
    // Only fields that semantically affect shared execution.

    auto operator<=>(const PackedCompatibilityKey&) const = default;
};
```

The exact fields must be determined from execution semantics, not copied blindly from the current comparison.

### Acceptance criteria

- one function defines compatibility identity;
- the key is part of or derived from the immutable execution plan;
- session compatibility compares keys plus explicit resource/device ownership identities;
- every compatibility-affecting option has a focused test;
- non-affecting observability or scheduling options do not cause false incompatibility;
- fingerprints and equality cannot drift independently.

---

## 10. C API CPU/CUDA composition is duplicated

The C API repeats backend selection, option conversion, service creation, and bundle wrapping. The real backend factory work should remove this duplication rather than hide it behind more local helper functions.

### Required direction

Fix the dependency boundary first. Keep ABI mapping functions focused on translating public C structs into neutral configuration values. Let the selected backend factory own concrete service construction.

### Acceptance criteria

- C API entry points remain small and backend-neutral;
- public struct-size and enum validation remains centralized;
- CPU-only builds do not reference CUDA implementation symbols;
- backend factories are independently testable;
- errors identify invalid ABI input separately from backend creation failure.

---

# SOLID-specific requirements

## Single Responsibility Principle

A valid SRP extraction must move at least one of:

- ownership;
- lifetime;
- dependency direction;
- invariant enforcement;
- state transition authority;
- compilation boundary;
- testable contract.

Moving methods to another `.cpp` while the original object still controls all state and decisions is not sufficient.

Pay special attention to:

- `PackedDecodeExecutorImpl`;
- generic runtime bootstrap;
- family architecture resolvers;
- central chat-template code;
- C API engine construction;
- GEMM dispatch ownership and caches.

## Open/Closed Principle

Adding a model family should not require edits across generic runtime, text, backend, and serving layers. Adding a backend should not require a new C API construction branch.

Closed domains are allowed when explicitly declared and tested. Do not mislabel adding another `switch` case as OCP compliance.

## Liskov Substitution Principle

- optional behavior belongs in separate capability interfaces;
- implementations must not inherit methods that are invalid for their storage/backend model;
- asynchronous completion contracts must define when state is observable;
- borrowed pointers and views must have documented owner lifetimes;
- backend factories that claim support must be able to create a usable service or return a defined unsupported result before partial mutation;
- family modules must be substitutable in a custom runtime builder without relying on hidden built-in globals.

## Interface Segregation Principle

Keep interfaces narrow and role-oriented. Do not replace current segregated repository or serving interfaces with composite god interfaces.

A family without vision, tools, native storage, or random access must not implement placeholder methods for those capabilities.

## Dependency Inversion Principle

Generic layers depend on neutral contracts. Concrete model families, backend services, checkpoint formats, and CUDA types must point inward through registration, compilation, or factory boundaries.

The following dependency directions are forbidden:

```text
src/runtime/**        -> celeg/models/**
include/celeg/model/** -> celeg/backend/cuda/**
src/backend/**        -> architecture identity dispatch
src/backend/**        -> concrete checkpoint format dispatch
src/api/**            -> concrete inference service construction
```

The C API may translate public ABI types, but service creation must be delegated through neutral factories.

---

# Implementation stages

Complete stages in this order unless current code evidence proves a dependency requires a different order. Keep each stage independently buildable and testable.

## Stage 0 — Re-baseline and protect behavior

1. Record the current commit and inspect changes since the reviewed baseline.
2. Run architecture boundary checks.
3. Build the available CPU and CUDA targets.
4. Run the configured tests.
5. Capture representative prompt rendering fixtures for all supported families.
6. Capture packed/non-packed and standard/ragged numerical parity where hardware is available.
7. Record steady-state allocation and representative performance baselines.

Do not claim runtime validation for CUDA tests that were only compiled.

## Stage 1 — Shared low-risk DRY utilities

Implement and test:

- JSON escaping;
- tagged-block scanning/removal;
- common metadata access helpers only where semantics are identical;
- common packed batch validation;
- immutable packed compatibility identity.

This stage should be behavior-preserving and should not move family registration yet.

## Stage 2 — Dense decoder topology builder

Extract shared dense topology invariants from MiniCPM5 and SmolLM3 while keeping probe rules and exceptional policies local. Add direct builder tests plus independent family-resolution tests.

Do not force LFM2, Granite, or Gemma4 through this builder unless their semantics fit naturally without family flags.

## Stage 3 — Family-owned chat/template/tool modules

Move family-specific templates and codecs into family directories. Remove duplicate tool rendering ownership. Preserve exact prompt fixtures.

## Stage 4 — Runtime family modules

Introduce the family registration boundary and remove concrete family knowledge from generic runtime bootstrap. Add static built-in modules and custom-runtime tests.

## Stage 5 — Real backend factories

Make backend factories create services. Remove concrete backend service construction from the C API and other generic composition roots.

## Stage 6 — Packed execution ownership

Move operation-specific orchestration into real decode and prefill pipeline owners. Keep shared fixed-capacity state explicit and hot-path-safe. Add failure-injection and lifecycle tests.

## Stage 7 — Expand automated architecture and DRY guards

Extend `scripts/check_architecture_boundaries.py` or add focused companion checks to detect:

- any generic runtime include of `celeg/models/`;
- concrete backend service construction from `src/api`;
- family names/types in generic runtime/text registration layers;
- duplicated local JSON escaping implementations;
- direct architecture or checkpoint-format dispatch in backends;
- plan compilation or direct CUDA allocation in packed hot paths;
- pass-through packed pipelines that delegate complete workflows back to shared state;
- obsolete architecture interfaces or compatibility paths.

Use static checks only for stable structural rules. Do not create fragile regex checks for ordinary implementation details.

## Stage 8 — Full verification and documentation

Run the required verification matrix, update architecture evidence, and replace stale plan statements with links to final implementation and executable tests.

---

# Testing requirements

At minimum, preserve or add coverage for:

## Architecture and dependency boundaries

- architecture catalog duplicate detection and freezing;
- custom architecture injection;
- custom runtime modules with and without built-ins;
- fake neutral repository compiling through CPU and CUDA compilers;
- no family-specific dependencies in generic runtime;
- no architecture/checkpoint-format dispatch in backends.

## Dense topology

- GGUF and SafeTensors/JSON key mapping;
- token defaults and overrides;
- multiple EOS tokens;
- attention slot/layer mappings;
- per-layer positional encoding exceptions;
- topology validation and fingerprint stability;
- MiniCPM5 and SmolLM3 independent resolution.

## Chat and tools

- exact prompt fixtures for every supported family;
- JSON escaping of quotes, slashes, controls, UTF-8, and empty strings;
- zero, one, and parallel tool calls;
- assistant text before, between, and after calls;
- complete, incomplete, and malformed tagged blocks;
- tool results and role capability enforcement;
- no duplicate tool-definition rendering.

## Backend factories and C API

- CPU factory creation;
- CUDA factory creation when available;
- unsupported CUDA build behavior;
- invalid backend and option mapping;
- factory failure diagnostics;
- C API smoke tests without concrete-service dependencies.

## Packed execution

- empty, oversized, null, duplicate, wrong-device, wrong-plan, wrong-storage, and incompatible-option batches;
- decode and prefill common validation consistency;
- packed vs non-packed decode parity;
- ragged vs standard prefill parity;
- mixed finalize and zero-finalize behavior;
- non-equal query width;
- variable per-layer FFN width;
- metadata storage replacement/rebinding;
- failure before host commit;
- completion and metrics boundaries;
- zero steady-state host/device allocations;
- representative performance regression thresholds.

---

# Performance constraints

Do not introduce:

- runtime architecture probing inside decode, prefill, attention, FFN, sampling, or scheduler loops;
- deep virtual object graphs in hot paths;
- per-token or per-layer heap allocation;
- repeated string-based tensor lookup during execution;
- repeated compatibility-key construction per layer;
- device capability caches that ignore device identity;
- synchronization added solely to simplify ownership unless existing correctness requires it and the performance cost is measured.

Prefer:

- compile-time or model-load-time registries;
- immutable plans and compatibility keys;
- direct calls, compact tagged records, function pointers, or templates after compilation;
- preallocated buffers;
- RAII for CUDA resources and scoped fan-out state;
- explicit event-based completion;
- measured changes with before/after artifacts.

Any material performance regression must be explained and either fixed or explicitly rejected. File count and line count are not success metrics.

---

# Required verification commands

Adapt paths to the active build environment, but run the repository's actual supported equivalents:

```text
python scripts/check_architecture_boundaries.py --root .
python scripts/dev.py verify
ctest --test-dir <configured-build-dir> --output-on-failure
```

Also run:

- CPU target build;
- CUDA target build when a CUDA toolchain is available;
- CUDA runtime tests only when a usable device is available;
- representative benchmark manifests before and after hot-path changes;
- `git diff --check`;
- any formatter or lint command already established by the repository.

If the full wrapper cannot finish, report the exact incomplete command and still run focused builds/tests. Never present compilation as runtime validation.

---

# Commit and working rules

- Work on the current target branch unless explicitly instructed otherwise.
- Do not create a pull request as part of this task.
- Use small, coherent commits whose messages describe the architectural outcome.
- Do not mix unrelated performance tuning with structural refactoring.
- Add regression tests before changing behavior for confirmed defects.
- Do not preserve dead compatibility paths without evidence that they are public and required.
- Do not add speculative abstractions for hypothetical families or backends.
- Do not stop after updating documentation; implement the code and tests.
- Do not declare a phase complete merely because code moved into another file.

---

# Final deliverable

When implementation is complete, provide:

1. the resulting commit range;
2. a concise mapping of each finding to changed files and tests;
3. before/after SOLID and DRY assessment with evidence;
4. commands run and exact results;
5. benchmarks and allocation evidence for hot-path changes;
6. any remaining debt, separated into correctness, architecture, DRY, performance, and optional enhancement categories;
7. explicit confirmation that generic runtime no longer knows concrete families, C API no longer constructs concrete backends, and packed pipelines own their workflows;
8. updated repository documentation pointing to final contracts and tests.

Do not claim completion for any item lacking source and test evidence.